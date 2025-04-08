from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II
from fairseq.dataclass import FairseqDataclass
from upstream.models.audio import D2vAudioConfig

@dataclass
class D2vPitchConfig(FairseqDataclass):
    extractor_mode: str = field(
        default="layer_norm",
        metadata={"help": "mode for feature extractor. default_norm or layer_norm"}
    )
    feature_encoder_spec: str = field(
        default="[(512, 10, 5)] + [(512, 8, 4)] + [(512, 4, 2)] + [(512, 4, 2)]",  # Điều chỉnh để gần với audio
        metadata={"help": "string describing convolutional feature extraction layers for pitch"}
    )

@dataclass
class D2vModalitiesTonalConfig(FairseqDataclass):
    audio: D2vAudioConfig = D2vAudioConfig()
    pitch: D2vPitchConfig = D2vPitchConfig()
    use_pitch: bool = field(default=True, metadata={"help": "whether to use the pitch branch"})
    pitch_proj_dim: Optional[int] = field(
        default=None, metadata={"help": "projection dim for pitch features, defaults to embed_dim"}
    )

@dataclass
class Data2VecMultiTonalConfig(FairseqDataclass):
    loss_beta: float = field(default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"})
    loss_scale: Optional[float] = field(default=None, metadata={"help": "scale the reconstruction loss"})
    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    cross_attn_num_heads: int = field(default=12, metadata={"help": "number of heads for cross-attention"})
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    cross_attn_dropout: float = field(default=0.1, metadata={"help": "dropout for cross-attention"})
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False
    average_top_k_layers: int = field(default=8, metadata={"help": "how many layers to average"})
    end_of_block_targets: bool = False
    clone_batch: int = 1
    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(default=0.9999, metadata={"help": "final ema decay rate"})
    ema_anneal_end_step: int = II("optimization.max_update")
    ema_encoder_only: bool = field(default=True, metadata={"help": "whether to update only encoder"})
    max_update: int = II("optimization.max_update")
    min_target_var: float = 0.1
    min_pred_var: float = 0.01
    modalities: D2vModalitiesTonalConfig = D2vModalitiesTonalConfig()
    type_embedding_dim: Optional[int] = field(default=None, metadata={"help": "dim for type embeddings"})
    num_embedding_types: int = field(default=2, metadata={"help": "number of input types"})