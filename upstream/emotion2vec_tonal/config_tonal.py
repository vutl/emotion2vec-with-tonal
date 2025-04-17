from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II
from fairseq.dataclass import FairseqDataclass
from upstream.models.audio import D2vAudioConfig


# -------------------- 1. CẤU HÌNH PITCH -------------------------------
@dataclass
class D2vPitchConfig(FairseqDataclass):
    extractor_mode: str = field(
        default="layer_norm",
        metadata={"help": "chế độ feature extractor: default_norm / layer_norm"},
    )

    feature_encoder_spec: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]",
        metadata={
            "help": (
                "mô tả các tầng convolution trích đặc trưng pitch "
                "(cấu hình khớp stride tổng 320 để T_pitch == T_audio)"
            )
        },
    )
# ----------------------------------------------------------------------


# -------------------- 2. CẤU HÌNH MODALITIES --------------------------
@dataclass
class D2vModalitiesTonalConfig(FairseqDataclass):
    audio: D2vAudioConfig = D2vAudioConfig()
    pitch: D2vPitchConfig = D2vPitchConfig()
    use_pitch: bool = field(
        default=True, metadata={"help": "dùng hay không dùng nhánh pitch"}
    )
    pitch_proj_dim: Optional[int] = field(
        default=None,
        metadata={"help": "projection dim cho pitch, mặc định = embed_dim"},
    )


# -------------------- 3. CẤU HÌNH MODEL CHUNG -------------------------
@dataclass
class Data2VecMultiTonalConfig(FairseqDataclass):
    loss_beta: float = 0
    loss_scale: Optional[float] = None
    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    cross_attn_num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    cross_attn_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False
    average_top_k_layers: int = 8
    end_of_block_targets: bool = False
    clone_batch: int = 1

    # ---------- EMA & ổn định huấn luyện ----------
    ema_decay: float = 0.999
    ema_same_dtype: bool = True
    ema_end_decay: float = 0.9999
    ema_anneal_end_step: int = II("optimization.max_update")
    ema_encoder_only: bool = True

    max_update: int = II("optimization.max_update")
    min_target_var: float = 0.1
    min_pred_var: float = 0.01

    # ------------ modalities -----------------------
    modalities: D2vModalitiesTonalConfig = D2vModalitiesTonalConfig()

    # ------------ embedding type (tùy chọn) --------
    type_embedding_dim: Optional[int] = None
    num_embedding_types: int = 2
