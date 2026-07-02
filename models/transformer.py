"""Transformer model -- raw features in, per-stock score out.

Input : x of shape (batch, T_lookback, F_features) per stock
Output: a scalar score per stock. For the long-only top-N strategy this is the
        predicted next-day return; we rank stocks by it each day and buy the top N.

Architecture
------------
    Linear(F -> d_model)
    + learned positional encoding over the T time steps
    -> N x TransformerEncoderLayer (self-attention over the lookback window)
    -> last-step (or mean) pooling
    -> Linear head -> scalar

`output`:
    "linear" -> raw score, trained with MSE against the realized return (default)
    "tanh"   -> bounded position in [-1, 1] (for a DLSA-style policy later)

Attention runs over raw normalized features directly -- no factor-model
preprocessing (this is the key difference from upstream DLSA).
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
        pooling: str = "last",   # "last" | "mean"
        output: str = "linear",  # "linear" | "tanh"
    ):
        super().__init__()
        self.pooling = pooling
        self.output = output
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
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
        h = self.encoder(h)                          # (B, T, d_model)
        h = h[:, -1] if self.pooling == "last" else h.mean(dim=1)
        out = self.head(h).squeeze(-1)               # (B,)
        return torch.tanh(out) if self.output == "tanh" else out
