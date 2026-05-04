import torch
import torch.nn as nn

from models.model import TransformerClassifier


class FusionMLP(nn.Module):
    """
    Shallow MLP over concatenated encoder embeddings.

    Null embeddings (learned) substitute a missing modality at inference,
    so the same model handles bimodal input, video-only, and sensor-only.
    Modality dropout during training prevents the head from ignoring either stream.
    """

    def __init__(self, n_classes: int, d_emb: int = 64, dropout: float = 0.5):
        super().__init__()
        self.null_v = nn.Parameter(torch.zeros(d_emb))
        self.null_s = nn.Parameter(torch.zeros(d_emb))
        self.mlp    = nn.Sequential(
            nn.Linear(d_emb * 2, d_emb), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_emb, n_classes),
        )

    def forward(
        self,
        e_v: torch.Tensor,
        e_s: torch.Tensor,
        drop_v: torch.Tensor = None,
        drop_s: torch.Tensor = None,
    ) -> torch.Tensor:
        if drop_v is not None:
            e_v = torch.where(drop_v[:, None], self.null_v[None].expand_as(e_v), e_v)
        if drop_s is not None:
            e_s = torch.where(drop_s[:, None], self.null_s[None].expand_as(e_s), e_s)
        return self.mlp(torch.cat([e_v, e_s], dim=1))


class FusionTransformerClassifier(nn.Module):
    """
    Multimodal fusion combining sensor and video via concatenated encoder embeddings + MLP.

    Architecture:
    - Two independent TransformerClassifier backbones (sensor + video)
    - Extract encoder features from each modality
    - Pool with mean + max temporal pooling per modality
    - Concatenate pooled representations → fusion MLP head

    Input:  x_sensor: (B, T_s, 12), x_video: (B, T_v, 98)
    Output: (B, n_classes)
    """

    def __init__(
        self,
        n_classes: int,
        in_dim_sensor: int,
        in_dim_video: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        d_fusion: int = 64,
    ):
        super().__init__()

        # Backbone networks
        self.sensor_backbone = TransformerClassifier(
            n_classes=n_classes,
            in_dim=in_dim_sensor,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.video_backbone = TransformerClassifier(
            n_classes=n_classes,
            in_dim=in_dim_video,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Fusion head: concatenates 2 modalities × 2 pooling methods × d_model
        # Input: (B, 4*d_model) → output: (B, n_classes)
        fusion_in_dim = 4 * d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, d_fusion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, n_classes),
        )

    def forward(self, x_sensor: torch.Tensor, x_video: torch.Tensor) -> torch.Tensor:
        """Fuse sensor and video modalities.

        Args:
            x_sensor: (B, T_s, in_dim_sensor) sensor sequence
            x_video: (B, T_v, in_dim_video) video sequence

        Returns:
            (B, n_classes) logits
        """
        # Extract encoder outputs without pooling
        z_sensor = self.sensor_backbone.get_encoder_features(x_sensor)  # (B, T_s, d_model)
        z_video = self.video_backbone.get_encoder_features(x_video)      # (B, T_v, d_model)

        # Pool each modality: mean + max over time
        z_sensor_mean = z_sensor.mean(dim=1)  # (B, d_model)
        z_sensor_max = z_sensor.max(dim=1)[0]  # (B, d_model)

        z_video_mean = z_video.mean(dim=1)   # (B, d_model)
        z_video_max = z_video.max(dim=1)[0]   # (B, d_model)

        # Concatenate all pooled representations
        z_fused = torch.cat([z_sensor_mean, z_sensor_max, z_video_mean, z_video_max], dim=1)

        # Classification
        return self.fusion_head(z_fused)
