#!/usr/bin/env python3
"""Candidate-level m6A biology model with per-case observation calibration."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMetaFormerBlock(nn.Module):
    """The same dilated depthwise-convolution block family as the site model."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
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
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.norm1(x).transpose(1, 2)
        mixed = self.pwconv(self.dwconv(mixed)).transpose(1, 2)
        x = x + self.drop1(mixed)
        return x + self.mlp(self.norm2(x))


class CandidateSequenceEncoder(nn.Module):
    """Newly initialized 201-nt sequence encoder producing a d192 embedding."""

    def __init__(
        self,
        sequence_length: int = 201,
        vocab_size: int = 5,
        d_model: int = 192,
        n_layers: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.15,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if sequence_length % 2 != 1:
            raise ValueError("sequence_length must be odd")
        self.sequence_length = sequence_length
        self.center_index = sequence_length // 2
        self.output_dim = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        dilations = [1, 2, 4, 8, 16, 1, 2, 4]
        if n_layers > len(dilations):
            extra = [1, 2, 4, 8, 16] * math.ceil(
                (n_layers - len(dilations)) / 5
            )
            dilations.extend(extra)
        self.blocks = nn.ModuleList(
            [
                ConvMetaFormerBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=dilations[layer_index],
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for layer_index in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        if sequence_tokens.ndim != 2:
            raise ValueError("sequence_tokens must have shape [batch, length]")
        if sequence_tokens.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"found {sequence_tokens.shape[1]}"
            )
        hidden = self.token_embedding(sequence_tokens) + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        center = hidden[:, self.center_index]
        mean_pool = hidden.mean(dim=1)
        attention = torch.softmax(
            self.attention_pool(hidden).squeeze(-1), dim=1
        )
        attention_pool = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        return self.output_projection(
            torch.cat([center, attention_pool, mean_pool], dim=-1)
        )


class CandidateM6ACalibrationModel(nn.Module):
    """Shared biology logits followed by a centered scalar beta per case_id."""

    def __init__(
        self,
        num_studies: int,
        num_regions: int = 4,
        region_embedding_dim: int = 16,
        numeric_feature_dim: int = 3,
        hidden_dim: int = 128,
        biology_mlp_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 201,
        encoder_dim: int = 192,
        encoder_layers: int = 8,
        encoder_kernel_size: int = 7,
        encoder_dropout: float = 0.15,
        precomputed_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_studies < 2:
            raise ValueError("At least two studies are required to center beta")
        if biology_mlp_layers < 1:
            raise ValueError("biology_mlp_layers must be at least one")
        self.num_studies = int(num_studies)
        self.precomputed_embedding_dim = precomputed_embedding_dim
        if precomputed_embedding_dim is None:
            self.sequence_encoder = CandidateSequenceEncoder(
                sequence_length=sequence_length,
                d_model=encoder_dim,
                n_layers=encoder_layers,
                kernel_size=encoder_kernel_size,
                dropout=encoder_dropout,
            )
            sequence_feature_dim = encoder_dim
        else:
            self.sequence_encoder = None
            sequence_feature_dim = int(precomputed_embedding_dim)

        self.region_embedding = nn.Embedding(
            num_embeddings=num_regions,
            embedding_dim=region_embedding_dim,
        )
        input_dim = sequence_feature_dim + region_embedding_dim + numeric_feature_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(biology_mlp_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.biology_hidden = nn.Sequential(*layers)
        self.biology_output = nn.Linear(hidden_dim, 1)

        self.beta = nn.Embedding(num_studies, 1)
        nn.init.zeros_(self.beta.weight)

    def get_centered_beta(self) -> torch.Tensor:
        beta_raw = self.beta.weight.squeeze(-1)
        return beta_raw - beta_raw.mean()

    def beta_l2_loss(self, lambda_beta: float) -> torch.Tensor:
        return float(lambda_beta) * self.get_centered_beta().pow(2).mean()

    def encode_candidates(
        self,
        region_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        sequence_tokens: torch.Tensor | None = None,
        precomputed_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sequence_encoder is not None:
            if sequence_tokens is None or precomputed_embeddings is not None:
                raise ValueError(
                    "Sequence mode requires sequence_tokens and no precomputed embeddings"
                )
            sequence_features = self.sequence_encoder(sequence_tokens)
        else:
            if precomputed_embeddings is None or sequence_tokens is not None:
                raise ValueError(
                    "Embedding mode requires precomputed_embeddings and no sequence tokens"
                )
            sequence_features = precomputed_embeddings
        region_features = self.region_embedding(region_ids)
        candidate_features = torch.cat(
            [sequence_features, region_features, numeric_features], dim=-1
        )
        return self.biology_output(
            self.biology_hidden(candidate_features)
        ).squeeze(-1)

    def calibrate_logits(
        self,
        bio_logits: torch.Tensor,
        study_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centered_beta = self.get_centered_beta()
        selected_beta = centered_beta[study_indices]
        return bio_logits + selected_beta, selected_beta

    def forward_candidate(
        self,
        region_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        study_indices: torch.Tensor | None = None,
        sequence_tokens: torch.Tensor | None = None,
        precomputed_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bio_logits = self.encode_candidates(
            sequence_tokens=sequence_tokens,
            precomputed_embeddings=precomputed_embeddings,
            region_ids=region_ids,
            numeric_features=numeric_features,
        )
        output = {
            "bio_logit": bio_logits,
            "q_biology": torch.sigmoid(bio_logits),
        }
        if study_indices is not None:
            obs_logits, selected_beta = self.calibrate_logits(
                bio_logits, study_indices
            )
            output.update(
                {
                    "beta_study": selected_beta,
                    "obs_logit": obs_logits,
                    "p_observed": torch.sigmoid(obs_logits),
                }
            )
        return output

    @staticmethod
    def forward_bag(
        obs_logits: torch.Tensor,
        instance_bag_indices: torch.Tensor,
        num_bags: int,
    ) -> dict[str, torch.Tensor]:
        """Numerically stable noisy-OR pooling over flattened bag instances."""
        obs_logits = obs_logits.float()
        log_not_prob_instance = F.logsigmoid(-obs_logits)
        log_p_none = torch.zeros(
            num_bags,
            dtype=obs_logits.dtype,
            device=obs_logits.device,
        )
        log_p_none.scatter_add_(
            0,
            instance_bag_indices,
            log_not_prob_instance,
        )
        p_bag = -torch.expm1(log_p_none)
        return {
            "log_p_none": log_p_none,
            "p_bag": p_bag.clamp(0.0, 1.0),
        }

    @staticmethod
    def bag_negative_log_likelihood(
        log_p_none: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Stable BCE for noisy-OR bags, returned without reduction."""
        labels = labels.to(log_p_none.dtype)
        # A positive bag can initially assign virtually zero probability to
        # every instance. Clamp that boundary before log1mexp: this avoids
        # infinite gradients without changing probabilities above 1e-7.
        min_positive_probability = 1e-7
        max_log_p_none = math.log1p(-min_positive_probability)
        safe_log_p_none = log_p_none.clamp_max(max_log_p_none)
        log_p_at_least_one = torch.log(-torch.expm1(safe_log_p_none))
        return torch.where(
            labels > 0.5,
            -log_p_at_least_one,
            -log_p_none,
        )
