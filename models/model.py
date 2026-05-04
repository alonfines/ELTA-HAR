import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.

    Input:  (B, T, in_dim)
    Output: (B, n_classes)
    """

    def __init__(
        self,
        n_classes: int,
        in_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj    = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.proj(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x_mean = x.mean(dim=1)
        x_max, _ = x.max(dim=1)
        return self.head(x_mean + x_max)

    def get_encoder_features(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Extract encoder outputs (B, T, d_model) before pooling/classification.

        Used by FusionTransformerClassifier to access intermediate token representations.
        """
        x = self.proj(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
