import torch
import torch.nn as nn
from typing import Optional

from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import TransposeLast, LayerNorm
from .config_tonal import D2vPitchConfig # Use the new pitch config

class PitchEncoder(nn.Module):
    def __init__(self, cfg: D2vPitchConfig, output_dim: Optional[int] = None):
        super().__init__()
        self.feature_enc_layers = eval(cfg.feature_encoder_spec)
        self.extractor_mode = cfg.extractor_mode
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=self.extractor_mode,
            conv_bias=True,
        )
        cnn_feature_dim = self.feature_enc_layers[-1][0]
        if output_dim is not None and output_dim != cnn_feature_dim:
            self.project_features = nn.Sequential(
                TransposeLast(),
                LayerNorm(cnn_feature_dim),
                nn.Linear(cnn_feature_dim, output_dim),
            )
        else:
            self.project_features = nn.Sequential(
                TransposeLast(),
                LayerNorm(cnn_feature_dim),
            )

    def forward(self, pitch_features, padding_mask=None, target_length=None):
        if pitch_features.dim() == 3:
            pitch_features = pitch_features.transpose(1, 2)  # [B, D_in, T]
        cnn_output = self.feature_extractor(pitch_features)  # [B, C_out, T_out]
        projected_output = self.project_features(cnn_output)  # [B, T_out, D_out]

        # Điều chỉnh độ dài sequence nếu cần
        if target_length is not None and projected_output.size(1) != target_length:
            projected_output = F.interpolate(
                projected_output.transpose(1, 2), 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)

        # Tính toán lại padding mask
        output_padding_mask = None
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            output_lengths = self.recalculate_padding_mask(input_lengths)
            batch_size, max_output_len = padding_mask.size(0), output_lengths.max()
            output_padding_mask = torch.zeros(
                (batch_size, max_output_len), dtype=torch.bool, device=pitch_features.device
            )
            for i in range(batch_size):
                output_padding_mask[i, output_lengths[i]:] = True
            if target_length is not None and max_output_len != target_length:
                output_padding_mask = F.interpolate(
                    output_padding_mask.float().unsqueeze(0), 
                    size=target_length, 
                    mode='nearest'
                ).squeeze(0).bool()

        return projected_output, output_padding_mask

    def recalculate_padding_mask(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)
        output_lengths = input_lengths
        for _, kernel_size, stride in self.feature_enc_layers:
            output_lengths = _conv_out_length(output_lengths, kernel_size, stride)
        return output_lengths.to(torch.long)

    def get_output_padding_mask(self, input_mask):
        """ Calculates the padding mask for the output of the CNN layers """

        if input_mask is None:
            return None

        input_lengths = (1 - input_mask.long()).sum(-1)
        output_lengths = self.recalculate_padding_mask(input_lengths)

        batch_size, max_output_len = input_mask.size(0), output_lengths.max() # Need max T_out
        # Reconstruct the mask based on output_lengths
        output_mask = torch.zeros((batch_size, max_output_len), dtype=torch.bool, device=input_mask.device)
        for i in range(batch_size):
            output_mask[i, output_lengths[i]:] = True # Mask positions beyond the calculated length

        return output_mask

