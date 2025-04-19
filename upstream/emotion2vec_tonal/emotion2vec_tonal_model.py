import copy, logging, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from upstream.models import BaseFairseqModel, register_model
from upstream.models.audio import AudioEncoder
from upstream.models.modules import AltBlock
from .config_tonal import Data2VecMultiTonalConfig
from .pitch_encoder import PitchEncoder

logger = logging.getLogger(__name__)

@register_model("data2vec_multi_tonal_crossattn", dataclass=Data2VecMultiTonalConfig)
class Data2VecMultiTonalCrossAttnModel(BaseFairseqModel):
    """
    emotion2vec with tonal fusion:
      1) AudioEncoder + PitchEncoder → features
      2) Cross‑attn A→P & P→A
      3) Concat → Self‑Attn → Linear(2D→D) = fused_repr
      4) (Student only) Mask fused_repr
      5) Backbone (AltBlock×depth)
      6) Teacher: same pipeline without mask
      7) Loss = frame‑level + utterance‑level
    """
    def __init__(self, cfg: Data2VecMultiTonalConfig, task=None):
        super().__init__()
        self.cfg = cfg
        self.task = task
        LN = partial(nn.LayerNorm, eps=cfg.norm_eps)

        # AudioEncoder (giữ y hệt emotion2vec gốc)
        def make_block(dp):
            return AltBlock(
                cfg.embed_dim, cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                norm_layer=LN, layer_norm_first=False
            )
        self.audio_enc = AudioEncoder(
            cfg.modalities.audio, cfg.embed_dim,
            lambda dp: make_block(dp), LN,
            layer_norm_first=False,
            alibi_biases={}, task=task
        )

        # PitchEncoder
        self.pitch_enc = PitchEncoder(cfg.modalities.pitch, output_dim=cfg.embed_dim)

        # Cross‑Attention A→P & P→A
        self.cross_ap = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
            dropout=cfg.cross_attn_dropout, batch_first=True
        )
        self.cross_pa = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
            dropout=cfg.cross_attn_dropout, batch_first=True
        )
        self.ln_q = LN(cfg.embed_dim)
        self.ln_kv = LN(cfg.embed_dim)
        self.drop_cross = nn.Dropout(cfg.cross_attn_dropout)

        # Fusion Self‑Attention & projection
        self.ln_f1 = LN(cfg.embed_dim*2)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim*2, num_heads=cfg.cross_attn_num_heads,
            dropout=cfg.attention_dropout, batch_first=True
        )
        self.ln_f2 = LN(cfg.embed_dim*2)
        self.proj_f = nn.Linear(cfg.embed_dim*2, cfg.embed_dim)

        # Mask token for student
        self.mask_emb = nn.Parameter(torch.zeros(1,1,cfg.embed_dim))
        nn.init.normal_(self.mask_emb, 0, 0.02)

        # Backbone (AltBlock × depth)
        self.blocks = nn.ModuleList([ make_block(0.0) for _ in range(cfg.depth) ])
        self.final_ln = LN(cfg.embed_dim)

        # Teacher = deepcopy, freeze
        self.teacher = copy.deepcopy(self)
        for p in self.teacher.parameters(): p.requires_grad_(False)
        if hasattr(self.teacher, "teacher"): delattr(self.teacher, "teacher")

        # Init
        self.apply(init_bert_params)

    def _encode_audio(self, wav, pad):
        x = self.audio_enc.local_features(wav)
        x = self.audio_enc.project_features(x)
        if self.audio_enc.fixed_positional_encoder is not None:
            x = x + self.audio_enc.fixed_positional_encoder(x, pad)
        return x

    def _encode_pitch(self, pitch, pad):
        xp, pad_p = self.pitch_enc(pitch, pad)
        return xp, pad_p

    @torch.no_grad()
    def update_teacher(self, tau=0.999):
        for s, t in zip(self.parameters(), self.teacher.parameters()):
            t.data.mul_(tau).add_(s.data, alpha=1-tau)

    def forward(
        self, source_audio, source_pitch,
        padding_mask_audio=None, padding_mask_pitch=None,
        mask=True, features_only=False, **kw
    ):
        # Encode
        xa = self._encode_audio(source_audio, padding_mask_audio)    # (B,T,D)
        xp, pad_p = self._encode_pitch(source_pitch, padding_mask_pitch)  # (B,T,D)

        # align length if needed
        if xa.size(1)!=xp.size(1):
            T = xa.size(1)
            xp = F.interpolate(
                xp.transpose(1,2), size=T, mode='linear', align_corners=False
            ).transpose(1,2)
            pad_p = F.interpolate(
                pad_p.float().unsqueeze(0), size=T, mode='nearest'
            ).squeeze(0).bool()

        # Cross‑attention both ways
        q_a = self.ln_q(xa); kv_p = self.ln_kv(xp)
        a2p, _ = self.cross_ap(q_a, kv_p, kv_p, key_padding_mask=pad_p)
        a2p = self.drop_cross(a2p)

        q_p = self.ln_q(xp); kv_a = self.ln_kv(xa)
        p2a, _ = self.cross_pa(q_p, kv_a, kv_a, key_padding_mask=padding_mask_audio)
        p2a = self.drop_cross(p2a)

        # Concat → Self‑Attn → Linear
        fused = torch.cat([a2p, p2a], dim=-1)        # (B,T,2D)
        fused = self.ln_f1(fused)
        fused,_ = self.self_attn(fused, fused, fused)
        fused = self.ln_f2(fused)
        fused = self.proj_f(fused)                  # (B,T,D)

        # Student: mask AFTER fusion
        mask_bool = None
        fused_student = fused
        if mask:
            _, m_info = self.audio_enc.compute_mask(
                fused, padding_mask_audio, apply=False
            )
            mask_bool = m_info.mask.bool()         # (B,T)
            mask_tok = self.mask_emb.expand(fused.size(0), fused.size(1), fused.size(2))
            fused_student = torch.where(
                mask_bool.unsqueeze(-1), mask_tok, fused
            )

        # Backbone on student
        x = fused_student
        for blk in self.blocks:
            x,_ = blk(x, padding_mask=padding_mask_audio, alibi_bias=None)
        x = self.final_ln(x)

        if features_only:
            return {"x": x, "mask": mask_bool}

        # Teacher: same pipeline WITHOUT mask
        with torch.no_grad():
            # reuse on teacher
            t_a = xa; t_p = xp
            ta2p,_ = self.cross_ap(self.ln_q(t_a), self.ln_kv(t_p), self.ln_kv(t_p), key_padding_mask=pad_p)
            tp2a,_ = self.cross_pa(self.ln_q(t_p), self.ln_kv(t_a), self.ln_kv(t_a), key_padding_mask=padding_mask_audio)
            tf = torch.cat([ta2p, tp2a], dim=-1)
            tf,_ = self.self_attn(self.ln_f1(tf), self.ln_f1(tf), self.ln_f1(tf))
            tf = self.proj_f(self.ln_f2(tf))
            for blk in self.blocks:
                tf,_ = blk(tf, padding_mask=padding_mask_audio, alibi_bias=None)
            t_repr = self.final_ln(tf)

        # Loss
        loss_f = F.mse_loss(x, t_repr, reduction='none')
        loss_f = loss_f[mask_bool].mean() if mask else loss_f.mean()
        loss_u = F.mse_loss(x.mean(1), t_repr.mean(1))
        loss = loss_f + loss_u

        return {"loss": loss, "student_x": x, "teacher_x": t_repr}

    def extract_features(
        self, source_audio, source_pitch=None,
        padding_mask_audio=None, padding_mask_pitch=None,
        mask=False, **kw
    ):
        out = self.forward(
            source_audio, source_pitch,
            padding_mask_audio, padding_mask_pitch,
            mask=mask, features_only=True, **kw
        )
        return out