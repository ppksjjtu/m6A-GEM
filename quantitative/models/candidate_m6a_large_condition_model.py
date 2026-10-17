#!/usr/bin/env python3
"""Large condition-aware candidate MIL model with multi-token AgroNT context."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.candidate_m6a_calibration_model import (
    CandidateM6ACalibrationModel,
    CandidateSequenceEncoder,
)


class CandidateM6ALargeConditionModel(nn.Module):
    """Predict condition-specific candidate biology and calibrate observations."""

    def __init__(
        self,
        *,
        num_studies: int,
        num_genotypes: int,
        num_conditions: int,
        condition_reference_id: int,
        num_protocols: int,
        qc_features: torch.Tensor,
        protocol_ids: torch.Tensor,
        gene_embedding_dim: int,
        sequence_length: int = 201,
        sequence_encoder_dim: int = 192,
        sequence_encoder_layers: int = 8,
        sequence_encoder_kernel_size: int = 7,
        sequence_encoder_dropout: float = 0.15,
        num_regions: int = 4,
        region_embedding_dim: int = 24,
        numeric_feature_dim: int = 3,
        hidden_dim: int = 384,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ffn_multiplier: int = 4,
        gene_token_count: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dim % transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if gene_token_count < 1:
            raise ValueError("gene_token_count must be positive")
        if qc_features.ndim != 2 or qc_features.shape[0] != num_studies:
            raise ValueError("qc_features must be [num_studies, num_qc_features]")
        if protocol_ids.shape != (num_studies,):
            raise ValueError("protocol_ids must contain one ID per study")

        self.num_studies = int(num_studies)
        self.num_genotypes = int(num_genotypes)
        self.num_conditions = int(num_conditions)
        self.hidden_dim = int(hidden_dim)
        self.gene_token_count = int(gene_token_count)

        # This encoder is deliberately initialized from scratch and remains
        # inside the end-to-end MIL graph. No pretrained candidate embedding is
        # loaded.
        self.sequence_encoder = CandidateSequenceEncoder(
            sequence_length=sequence_length,
            d_model=sequence_encoder_dim,
            n_layers=sequence_encoder_layers,
            kernel_size=sequence_encoder_kernel_size,
            dropout=sequence_encoder_dropout,
        )
        self.region_embedding = nn.Embedding(num_regions, region_embedding_dim)
        candidate_input_dim = (
            int(sequence_encoder_dim)
            + int(region_embedding_dim)
            + int(numeric_feature_dim)
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.genotype_embedding = nn.Embedding(
            num_genotypes, hidden_dim, padding_idx=0
        )
        nn.init.normal_(self.genotype_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.genotype_embedding.weight[0].zero_()

        self.condition_embedding = nn.Embedding(
            num_conditions,
            hidden_dim,
            padding_idx=condition_reference_id,
        )
        nn.init.zeros_(self.condition_embedding.weight)

        # The 1500-dimensional AgroNT vector is expanded into several memory
        # tokens instead of being squeezed through one narrow gene token.
        self.gene_projection = nn.Sequential(
            nn.LayerNorm(gene_embedding_dim),
            nn.Linear(gene_embedding_dim, gene_token_count * hidden_dim),
            nn.GELU(),
        )
        self.gene_token_type = nn.Parameter(
            torch.empty(gene_token_count, hidden_dim)
        )
        nn.init.normal_(self.gene_token_type, mean=0.0, std=0.02)
        self.gene_token_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # FiLM makes genotype/condition interact with sequence and AgroNT
        # features instead of acting only as an additive global offset.
        self.condition_film = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        nn.init.zeros_(self.condition_film[-1].weight)
        nn.init.zeros_(self.condition_film[-1].bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * transformer_ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.biology_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Head 2: a shared study/QC/protocol observation calibration. It has no
        # condition-specific free offset, so it cannot explain within-study
        # condition changes by itself.
        self.study_bias = nn.Embedding(num_studies, 1)
        self.qc_linear = nn.Linear(qc_features.shape[1], 1, bias=False)
        self.protocol_bias = nn.Embedding(num_protocols, 1)
        nn.init.zeros_(self.study_bias.weight)
        nn.init.zeros_(self.qc_linear.weight)
        nn.init.zeros_(self.protocol_bias.weight)
        self.register_buffer("qc_features", qc_features.float())
        self.register_buffer("protocol_ids", protocol_ids.long())

    def encode_unique_candidates(self, batch: dict) -> torch.Tensor:
        sequence_features = self.sequence_encoder(batch["sequence_tokens"])
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

    def _apply_conditioning(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.condition_film(conditioning).chunk(2, dim=-1)
        scale = 0.25 * torch.tanh(scale)
        return tokens * (1.0 + scale) + shift + conditioning

    def contextual_biology_outputs(
        self,
        batch: dict,
        precomputed_unique_local_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        unique_tokens = (
            self.encode_unique_candidates(batch)
            if precomputed_unique_local_hidden is None
            else precomputed_unique_local_hidden
        )
        if unique_tokens.ndim != 2 or unique_tokens.shape[1] != self.hidden_dim:
            raise ValueError(
                "precomputed unique local hidden states have an invalid shape"
            )
        context_local_tokens = unique_tokens[batch["context_to_unique"]]

        genotype = self.genotype_embedding(batch["context_genotype_ids"])
        condition = self.condition_embedding(batch["context_condition_ids"])
        conditioning = genotype + condition
        group_indices = batch["context_group_indices"]
        candidate_tokens = self._apply_conditioning(
            context_local_tokens,
            conditioning[group_indices],
        )

        group_lengths = batch["context_group_lengths"]
        num_groups = len(group_lengths)
        max_candidates = int(group_lengths.max())
        total_length = max_candidates + self.gene_token_count
        padded = candidate_tokens.new_zeros(
            (num_groups, total_length, self.hidden_dim)
        )
        padding_mask = torch.ones(
            (num_groups, total_length),
            dtype=torch.bool,
            device=candidate_tokens.device,
        )

        gene_tokens = self.gene_projection(batch["gene_embeddings"]).view(
            num_groups, self.gene_token_count, self.hidden_dim
        )
        gene_tokens = gene_tokens + self.gene_token_type.unsqueeze(0)
        gene_tokens = gene_tokens + self.gene_token_refiner(gene_tokens)
        gene_conditioning = conditioning.unsqueeze(1).expand_as(gene_tokens)
        gene_tokens = self._apply_conditioning(
            gene_tokens, gene_conditioning
        )
        padded[:, : self.gene_token_count] = gene_tokens
        gene_available = batch["gene_embedding_available"].bool()
        padding_mask[:, : self.gene_token_count] = ~gene_available.unsqueeze(1)

        candidate_positions = (
            batch["context_position_indices"] + self.gene_token_count
        )
        padded[group_indices, candidate_positions] = candidate_tokens
        padding_mask[group_indices, candidate_positions] = False

        contextualized = self.context_encoder(
            padded, src_key_padding_mask=padding_mask
        )
        candidate_context = contextualized[group_indices, candidate_positions]
        context_logits = self.biology_head(candidate_context).squeeze(-1)
        instance_indices = batch["instance_context_indices"]
        return {
            "instance_biology_logits": context_logits[instance_indices],
            "instance_context_hidden": candidate_context[instance_indices],
            "instance_local_hidden": context_local_tokens[instance_indices],
        }

    def contextual_biology_logits(self, batch: dict) -> torch.Tensor:
        return self.contextual_biology_outputs(batch)[
            "instance_biology_logits"
        ]

    @staticmethod
    def _center(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean()

    def calibration_components(self) -> dict[str, torch.Tensor]:
        study = self._center(self.study_bias.weight.squeeze(-1))
        qc = self._center(self.qc_linear(self.qc_features).squeeze(-1))
        protocol = self._center(
            self.protocol_bias(self.protocol_ids).squeeze(-1)
        )
        return {
            "study_bias": study,
            "qc_beta": qc,
            "protocol_beta": protocol,
            "beta_total": study + qc + protocol,
        }

    def calibrate_logits(
        self,
        bio_logits: torch.Tensor,
        study_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self.calibration_components()
        selected = {
            name: values[study_indices]
            for name, values in components.items()
        }
        return bio_logits + selected["beta_total"], selected

    forward_bag = staticmethod(CandidateM6ACalibrationModel.forward_bag)
    bag_negative_log_likelihood = staticmethod(
        CandidateM6ACalibrationModel.bag_negative_log_likelihood
    )
