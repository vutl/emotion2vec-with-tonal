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

    def __init__(self, cfg: Data2VecMultiTonalConfig, task=None):
        super().__init__()
        self.cfg  = cfg
        self.task = task
        ln = partial(nn.LayerNorm, eps=1e-6)

        # ===== AudioEncoder gốc từ emotion2vec ========================
        def make_block(dp):
            return AltBlock(
                cfg.embed_dim, cfg.num_heads,
                mlp_ratio      = cfg.mlp_ratio,
                drop           = cfg.encoder_dropout,
                attn_drop      = cfg.attention_dropout,
                mlp_drop       = cfg.activation_dropout,
                norm_layer     = ln,
                layer_norm_first=False
            )

        self.alibi_biases = {}                        # cache alibi gốc
        self.audio_enc    = AudioEncoder(
            cfg.modalities.audio, cfg.embed_dim,
            lambda dp: make_block(dp), ln,
            layer_norm_first=False,
            alibi_biases=self.alibi_biases,
            task=task
        )

        # ===== PitchEncoder ===========================================
        self.pitch_enc = PitchEncoder(
            cfg.modalities.pitch,
            output_dim = cfg.embed_dim
        )

        # ===== Cross‑Attention AUDIO ← PITCH ==========================
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = cfg.embed_dim,
            num_heads = cfg.num_heads,
            dropout   = cfg.cross_attn_dropout,
            batch_first=True
        )
        self.ln_q   = ln(cfg.embed_dim)
        self.ln_kv  = ln(cfg.embed_dim)
        self.ln_out = ln(cfg.embed_dim)
        self.drop_cross = nn.Dropout(cfg.cross_attn_dropout)

        # ===== 8 AltBlock backbone ====================================
        self.blocks = nn.ModuleList([make_block(0.0) for _ in range(cfg.depth)])
        self.final_ln = ln(cfg.embed_dim)

        # ===== mask token learnable ===================================
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.normal_(self.mask_emb, mean=0., std=0.02)

        # ===== Teacher (deepcopy, bỏ đệ‑quy, không train) ==============
        self.teacher = copy.deepcopy(self)           # clone toàn bộ
        if hasattr(self.teacher, "teacher"):         # tránh lồng nhau
            delattr(self.teacher, "teacher")
        for p in self.teacher.parameters():
            p.requires_grad_(False)                  # freeze teacher

        # ===== Khởi tạo style BERT ====================================
        self.apply(init_bert_params)

    # encode audio + posEnc
    def _encode_audio(self, wav, pad):
        x = self.audio_enc.local_features(wav)
        x = self.audio_enc.project_features(x)
        if self.audio_enc.fixed_positional_encoder is not None:
            x = x + self.audio_enc.fixed_positional_encoder(x, pad)
        return x                                         # (B,T,D)

    # EMA cập nhật teacher
    @torch.no_grad()
    def update_teacher(self, tau: float = 0.999):
        for s, t in zip(self.parameters(), self.teacher.parameters()):
            t.data.mul_(tau).add_(s.data, alpha=1 - tau)

    def forward(
        self,
        source_audio,
        source_pitch,
        padding_mask_audio=None,
        padding_mask_pitch=None,
        mask=True,
        features_only=False,
        **kwargs
    ):
        # -------- 1. Audio & Pitch embedding -------------------------
        xa = self._encode_audio(source_audio, padding_mask_audio)           # (B,T,D)
        xp, pad_p = self.pitch_enc(source_pitch, padding_mask_pitch)        # (B,Tp,D)

        #  (rare) khớp chiều dài nếu stride pitch ≠ stride audio
        if xa.size(1) != xp.size(1):
            T = xa.size(1)
            xp = F.interpolate(xp.transpose(1,2), size=T,
                                mode='linear', align_corners=False).transpose(1,2)

        # -------- 2. Mask audio cho STUDENT --------------------------
        mask_bool = None
        q_audio   = xa
        if mask:
            _, m_info = self.audio_enc.compute_mask(xa, padding_mask_audio,
                                                    apply=False)
            mask_bool = m_info.mask.bool()                # (B,T)
            q_audio   = torch.where(
                mask_bool.unsqueeze(-1),
                self.mask_emb.repeat(xa.size(0), xa.size(1), 1),
                xa
            )

        # -------- 3. Cross‑Attention AUDIO ← PITCH -------------------
        q = self.ln_q(q_audio)
        k = self.ln_kv(xp)
        ctx,_ = self.cross_attn(
            q, k, k,
            key_padding_mask = pad_p
        )
        x = self.ln_out(q_audio + self.drop_cross(ctx))    # (B,T,D)

        # -------- 4. Backbone Transformer ----------------------------
        for blk in self.blocks:
            x,_ = blk(x, padding_mask=padding_mask_audio, alibi_bias=None)
        x = self.final_ln(x)

        # -------- 5. Chỉ lấy feature--------------------------------
        if features_only:
            return {"x": x, "mask": mask_bool}

        # -------- 6. Teacher forward (không mask) --------------------
        with torch.no_grad():
            t = self.teacher._forward_no_mask(
                source_audio, source_pitch,
                padding_mask_audio, pad_p
            )                                             # (B,T,D)

        # -------- 7. Loss (frame ở mask + utterance) -----------------
        loss_f = F.mse_loss(x, t, reduction='none')
        if mask:
            loss_f = loss_f[mask_bool].mean()
        else:
            loss_f = loss_f.mean()
        loss_u = F.mse_loss(x.mean(1), t.mean(1))
        loss   = loss_f + loss_u

        return {"loss": loss, "student_x": x, "teacher_x": t}

    # ------------------------------------------------------------------ #
    #                TEACHER forward **KHÔNG mask**                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _forward_no_mask(self, wav, pitch, pad_a, pad_p):
        xa = self._encode_audio(wav, pad_a)
        xp,_ = self.pitch_enc(pitch, pad_p)
        if xa.size(1) != xp.size(1):
            xp = F.interpolate(xp.transpose(1,2), size=xa.size(1),
                               mode='linear', align_corners=False).transpose(1,2)
        q = self.ln_q(xa)
        k = self.ln_kv(xp)
        ctx,_ = self.cross_attn(q, k, k, key_padding_mask=pad_p)
        x = self.ln_out(xa + ctx)
        for blk in self.blocks:
            x,_ = blk(x, padding_mask=pad_a, alibi_bias=None)
        return self.final_ln(x)


    def extract_features(
        self,
        source_audio,
        source_pitch        = None,
        padding_mask_audio  = None,
        padding_mask_pitch  = None,
        mask                = False,
        **kwargs
    ):
        """
        Trả về dict:
          {
              'x'   : Tensor (B,T,D)  – embedding cuối,
              'mask': BoolTensor|None – vị trí mask audio (nếu mask=True)
          }
        """
        return self.forward(
            source_audio,
            source_pitch,
            padding_mask_audio,
            padding_mask_pitch,
            mask          = mask,
            features_only = True,
            **kwargs
        )

