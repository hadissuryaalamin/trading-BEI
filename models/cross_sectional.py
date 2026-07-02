"""Cross-sectional Transformer: temporal encoder per stock, then attention across stocks.

Two stages:
  1. Temporal: each stock's (lookback, F) window -> one d_model vector (self-attention
     over the 60 days, like the baseline but returning a vector instead of a scalar).
  2. Cross-sectional: on a given day, self-attention across ALL stocks so each stock's
     representation is informed by the whole market that day (no positional encoding --
     stocks are an unordered set). Then a head produces one score per stock.

forward expects ONE day at a time:
    x: (N_stocks, lookback, n_features)  ->  scores: (N_stocks,)
Rank the scores to pick the top-N to buy. This is the difference-maker vs the
per-stock baseline: relative comparison across the cross-section.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossSectionalModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        temporal_layers: int = 2,
        cross_layers: int = 2,
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

        t_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True)
        self.temporal = nn.TransformerEncoder(t_layer, temporal_layers)

        c_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True)
        self.cross = nn.TransformerEncoder(c_layer, cross_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (N, T, F) -- one day's N stocks
        h = self.input_proj(x) + self.pos_emb[:, : x.size(1)]
        h = self.temporal(h)                                   # (N, T, d)
        h = h[:, -1] if self.pooling == "last" else h.mean(dim=1)  # (N, d)

        h = h.unsqueeze(0)                                     # (1, N, d) -- N as sequence
        h = self.cross(h, src_key_padding_mask=key_padding_mask)
        out = self.head(h).squeeze(0).squeeze(-1)             # (N,)
        return torch.tanh(out) if self.output == "tanh" else out
