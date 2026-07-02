"""Transformer trading-policy model — raw features in, position out.

Input  : x of shape (batch, T_lookback, F_features) per asset
Output : a scalar position/weight for the next period (batch,)

Architecture (baseline)
-----------------------
    input Linear(F -> d_model)
    + positional encoding (learned or sinusoidal) over the T time steps
    -> N x TransformerEncoderLayer (self-attention over the lookback window)
    -> take last-step (or mean-pooled) token
    -> Linear head (d_model -> 1)  ->  tanh (bounded position)

Positions are combined across the cross-section at portfolio-construction time
(dollar-neutral / leverage constraint) in train_test.py, not inside the module.

This replaces the DLSA CNN+Transformer-on-residuals: here attention runs over
raw normalized features directly, with no factor-model preprocessing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
        lookback: int = 60,
        pooling: str = "last",  # "last" | "mean"
    ):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, lookback, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x) + self.pos_emb[:, : x.size(1)]
        h = self.encoder(h)                       # (B, T, d_model)
        h = h[:, -1] if self.pooling == "last" else h.mean(dim=1)
        return torch.tanh(self.head(h)).squeeze(-1)  # (B,) position in [-1, 1]
