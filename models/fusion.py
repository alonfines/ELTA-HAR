import torch
import torch.nn as nn


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
