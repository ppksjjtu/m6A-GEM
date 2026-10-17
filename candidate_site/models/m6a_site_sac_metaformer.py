#!/usr/bin/env python3
"""Sequence-only m6A A-site model for Atlas/RMBase + SAC-seq training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


PAD_TOKEN = 5


class ConvMetaFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 9,
        dilation: int = 1,
        dropout: float = 0.15,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
        )
        self.pwconv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x).transpose(1, 2)
        h = self.pwconv(self.dwconv(h)).transpose(1, 2)
        x = x + self.drop1(h)
        x = x + self.mlp(self.norm2(x))
        return x * input_mask.unsqueeze(-1)


class M6ASiteSACMetaFormer(nn.Module):
    """Predict every A in a variable-length peak core with flanking context."""

    def __init__(
        self,
        seq_len: int = 1400,
        vocab_size: int = 6,
        d_model: int = 192,
        n_layers: int = 13,
        kernel_size: int = 9,
        dropout: float = 0.15,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.peak_region_emb = nn.Embedding(2, d_model)

        base_dilations = [1, 2, 4, 8, 16, 32, 64]
        dilations = (base_dilations * math.ceil(n_layers / len(base_dilations)))[
            :n_layers
        ]
        self.blocks = nn.ModuleList(
            [
                ConvMetaFormerBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=dilations[index],
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.site_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.window_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.window_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq: torch.Tensor,
        input_mask: torch.Tensor,
        peak_mask: torch.Tensor,
    ):
        input_mask = input_mask.bool()
        peak_mask = peak_mask.bool()
        x = self.token_emb(seq) + self.peak_region_emb(peak_mask.long())
        x = x * input_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, input_mask)
        hidden = self.final_norm(x) * input_mask.unsqueeze(-1)

        site_logits = self.site_head(hidden).squeeze(-1)
        attention_logits = self.window_pool(hidden).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~peak_mask, -1e4)
        attention = torch.softmax(attention_logits, dim=1)
        attention_pool = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        denominator = peak_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean_pool = (hidden * peak_mask.unsqueeze(-1)).sum(dim=1) / denominator
        window_logit = self.window_head(
            torch.cat([attention_pool, mean_pool], dim=-1)
        ).squeeze(-1)
        return site_logits, window_logit, hidden, attention


def build_model_from_config(config: dict) -> M6ASiteSACMetaFormer:
    return M6ASiteSACMetaFormer(
        seq_len=int(config.get("seq_len", 1400)),
        d_model=int(config.get("d_model", 192)),
        n_layers=int(config.get("n_layers", 13)),
        kernel_size=int(config.get("kernel_size", 9)),
        dropout=float(config.get("dropout", 0.15)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
    )
