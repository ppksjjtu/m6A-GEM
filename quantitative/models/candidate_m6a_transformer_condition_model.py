#!/usr/bin/env python3
"""State1 condition-aware MIL model with a pure Transformer sequence encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.candidate_m6a_large_condition_model import (
    CandidateM6ALargeConditionModel as _BaseConditionModel,
)


class CandidateTransformerSequenceEncoder(nn.Module):
    """Encode a 201-nt candidate window without convolutional layers."""

    def __init__(
        self,
        *,
        sequence_length: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        n_heads: int = 8,
        ffn_multiplier: int = 4,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.sequence_length = int(sequence_length)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.base_embedding = nn.Embedding(5, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.empty(1, sequence_length + 1, d_model)
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * ffn_multiplier,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.base_embedding.weight, mean=0.0, std=0.02)

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        if sequence_tokens.ndim != 2:
            raise ValueError("sequence_tokens must have shape [batch, length]")
        if sequence_tokens.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected sequence length {self.sequence_length}, "
                f"received {sequence_tokens.shape[1]}"
            )
        batch_size = sequence_tokens.shape[0]
        bases = self.base_embedding(sequence_tokens.long())
        cls = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, bases], dim=1)
        hidden = self.input_dropout(
            self.input_norm(hidden + self.position_embedding)
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        hidden = self.output_norm(hidden)
        cls_state = hidden[:, 0]
        center_state = hidden[:, 1 + self.sequence_length // 2]
        mean_state = hidden[:, 1:].mean(dim=1)
        return self.pool_projection(
            torch.cat([cls_state, center_state, mean_state], dim=-1)
        )


class CandidateM6ATransformerConditionModel(_BaseConditionModel):
    """Keep State1 context/MIL/calibration heads and replace only sequence CNN."""

    def __init__(
        self,
        *,
        sequence_chunk_size: int = 128,
        sequence_transformer_heads: int = 8,
        sequence_transformer_ffn_multiplier: int = 4,
        sequence_gradient_checkpointing: bool = True,
        **kwargs,
    ) -> None:
        sequence_length = int(kwargs.get("sequence_length", 201))
        sequence_encoder_dim = int(kwargs.get("sequence_encoder_dim", 192))
        sequence_encoder_layers = int(kwargs.get("sequence_encoder_layers", 6))
        sequence_encoder_dropout = float(
            kwargs.get("sequence_encoder_dropout", 0.15)
        )
        super().__init__(**kwargs)
        self.sequence_chunk_size = int(sequence_chunk_size)
        self.sequence_encoder = CandidateTransformerSequenceEncoder(
            sequence_length=sequence_length,
            d_model=sequence_encoder_dim,
            n_layers=sequence_encoder_layers,
            dropout=sequence_encoder_dropout,
            n_heads=sequence_transformer_heads,
            ffn_multiplier=sequence_transformer_ffn_multiplier,
            gradient_checkpointing=sequence_gradient_checkpointing,
        )

    def encode_unique_candidates(self, batch: dict) -> torch.Tensor:
        sequences = batch["sequence_tokens"]
        chunk_size = max(1, self.sequence_chunk_size)
        sequence_features = torch.cat(
            [
                self.sequence_encoder(sequences[start : start + chunk_size])
                for start in range(0, sequences.shape[0], chunk_size)
            ],
            dim=0,
        )
        return self.candidate_projection(
            torch.cat(
                [
                    sequence_features,
                    self.region_embedding(batch["region_ids"]),
                    batch["numeric_features"],
                ],
                dim=-1,
            )
        )


# The existing trainer constructs this symbol. Aliasing lets us reuse the exact
# State1 data, MIL objective, condition context, calibration, and evaluation.
CandidateM6ALargeConditionModel = CandidateM6ATransformerConditionModel
