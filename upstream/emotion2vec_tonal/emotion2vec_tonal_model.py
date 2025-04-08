# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from upstream.models import BaseFairseqModel, register_model
from upstream.models.modules import AltBlock, AltAttention
from upstream.models.audio import AudioEncoder
from upstream.models.base import MaskInfoaaa

from .config_tonal import Data2VecMultiTonalConfig
from .pitch_encoder import PitchEncoder

logger = logging.getLogger(__name__)

@register_model("data2vec_multi_tonal_crossattn", dataclass=Data2VecMultiTonalConfig)
class Data2VecMultiTonalCrossAttnModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecMultiTonalConfig, task=None):
        super().__init__()
        self.cfg = cfg
        self.task = task

        make_layer_norm = partial(nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine)

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()

        # Audio branch: sử dụng AudioEncoder gốc
        audio_mod_cfg = getattr(cfg.modalities, 'audio')
        self.audio_encoder = AudioEncoder(audio_mod_cfg, cfg.embed_dim, make_block, make_layer_norm, cfg.layer_norm_first, self.alibi_biases, task)
        self.modality_encoders['AUDIO'] = self.audio_encoder

        # Pitch branch: khởi tạo PitchEncoder nếu use_pitch=True
        self.pitch_encoder = None
        if cfg.modalities.use_pitch:
            pitch_mod_cfg = getattr(cfg.modalities, 'pitch')
            pitch_proj_dim = cfg.embed_dim  # đảm bảo kích thước phù hợp cho cross-attention
            self.pitch_encoder = PitchEncoder(pitch_mod_cfg, output_dim=pitch_proj_dim)

        # Cross-attention để hợp nhất thông tin giữa audio và pitch
        self.audio_pitch_cross_attention = None
        if cfg.modalities.use_pitch:
            self.audio_pitch_cross_attention = nn.MultiheadAttention(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,  # dùng num_heads cho cross-attn
                dropout=cfg.attention_dropout,
                batch_first=True,
                bias=True,
            )
            self.norm_cross_attn_query = make_layer_norm(cfg.embed_dim)
            self.norm_cross_attn_kv = make_layer_norm(cfg.embed_dim)
            self.dropout_cross_attn = nn.Dropout(cfg.attention_dropout)
            self.norm_after_cross_attn = make_layer_norm(cfg.embed_dim)

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.normal_(self.mask_embedding, mean=0, std=0.02)

        # Teacher-student: khởi tạo bản sao teacher của mô hình hiện tại
        # (Lưu ý: trong thực tế bạn có thể muốn tách riêng teacher, nhưng dưới đây là một ví dụ đơn giản)
        self.teacher_model = copy.deepcopy(self)
        for p in self.teacher_model.parameters():
            p.requires_grad = False  # Teacher không cập nhật qua backprop

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale
        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)
        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        self.apply(init_bert_params)
        if hasattr(self.audio_encoder, 'reset_parameters'):
            self.audio_encoder.reset_parameters()
        if self.pitch_encoder is not None and hasattr(self.pitch_encoder, 'reset_parameters'):
            self.pitch_encoder.reset_parameters()

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

    def update_teacher(self, tau: float):
        """
        Cập nhật teacher bằng EMA từ student.
        tau: hệ số EMA (ví dụ: 0.999)
        """
        with torch.no_grad():
            for student_param, teacher_param in zip(self.parameters(), self.teacher_model.parameters()):
                teacher_param.data.mul_(tau).add_(student_param.data, alpha=1 - tau)

    def forward(self, source_audio, source_pitch=None, padding_mask_audio=None, padding_mask_pitch=None, mask=True, features_only=False, **kwargs):
        # Lấy embedding từ audio branch
        x_audio = self.audio_encoder.local_features(source_audio)
        x_audio = self.audio_encoder.project_features(x_audio)
        if self.audio_encoder.fixed_positional_encoder is not None:
            x_audio = x_audio + self.audio_encoder.fixed_positional_encoder(x_audio, padding_mask_audio)

        # Lấy embedding từ pitch branch nếu có
        x_pitch = None
        if self.cfg.modalities.use_pitch and source_pitch is not None and self.pitch_encoder is not None:
            x_pitch, _ = self.pitch_encoder(source_pitch, padding_mask_pitch)
            if x_audio.size(1) != x_pitch.size(1):
                target_len = x_audio.size(1)
                x_pitch = F.interpolate(x_pitch.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
                if padding_mask_pitch is not None:
                    padding_mask_pitch = F.interpolate(padding_mask_pitch.float().unsqueeze(0), size=target_len, mode='nearest').squeeze(0).bool()

        # Xử lý mask cho audio branch: nếu mask==True, thay các token bị mask bằng mask_embedding
        audio_query = x_audio
        audio_mask_bool = None
        if mask:
            # Sử dụng hàm compute_mask của audio_encoder để lấy thông tin mask
            _, mask_info_audio = self.audio_encoder.compute_mask(x_audio, padding_mask_audio, mask_seed=kwargs.get("mask_seeds", None), apply=False)
            audio_mask_bool = mask_info_audio.mask.bool()
            mask_emb = self.mask_embedding.repeat(x_audio.size(0), x_audio.size(1), 1)
            audio_query = torch.where(audio_mask_bool.unsqueeze(-1), mask_emb, x_audio)

        # Cross-attention: hợp nhất thông tin từ audio và pitch
        if self.audio_pitch_cross_attention is not None and x_pitch is not None:
            q = self.norm_cross_attn_query(audio_query)
            k = self.norm_cross_attn_kv(x_pitch)
            v = k
            cross_attn_key_padding_mask = padding_mask_pitch
            context_from_pitch, _ = self.audio_pitch_cross_attention(
                query=q, key=k, value=v, key_padding_mask=cross_attn_key_padding_mask
            )
            x_combined = audio_query + self.dropout_cross_attn(context_from_pitch)
            x_combined = self.norm_after_cross_attn(x_combined)
        else:
            x_combined = audio_query

        if self.audio_encoder.relative_positional_encoder is not None:
            x_combined = x_combined + self.audio_encoder.relative_positional_encoder(x_combined)

        x_combined = self.dropout_input(x_combined)
        layer_results = []
        current_padding_mask = padding_mask_audio
        current_alibi_bias = None

        for i, blk in enumerate(self.blocks):
            if not self.training or self.cfg.layerdrop == 0 or (np.random.random() > self.cfg.layerdrop):
                x_combined, lr = blk(x_combined, padding_mask=current_padding_mask, alibi_bias=current_alibi_bias)
                layer_results.append(lr)
        if self.norm is not None:
            x_combined = self.norm(x_combined)

        if features_only:
            return {"x": x_combined, "padding_mask": padding_mask_audio, "mask": audio_mask_bool}
        else:
            return {
                "final_output": x_combined,
                "padding_mask_audio": padding_mask_audio,
                "audio_mask": audio_mask_bool,
                "layer_results": layer_results,
            }

    def extract_features(self, source_audio, source_pitch=None, padding_mask_audio=None, padding_mask_pitch=None, mask=False, **kwargs):
        return self.forward(source_audio, source_pitch, padding_mask_audio, padding_mask_pitch, mask, features_only=True, **kwargs)
