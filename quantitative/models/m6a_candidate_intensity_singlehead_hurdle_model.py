#!/usr/bin/env python3
"""Single direct-intensity hurdle model for the Copper-normal experiment.

There is no biology-head/observation-head cascade and no case calibration.
The inherited scalar ``biology_head`` is used directly as the sole positive
intensity head. A separate presence head is retained because the hurdle
objective requires P(I > 0). Inference exposes both the continuous expected
intensity and an optional thresholded sparse intensity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.candidate_m6a_transformer_condition_model import (
    CandidateM6ATransformerConditionModel,
)


class CandidateIntensitySingleHeadHurdleModel(
    CandidateM6ATransformerConditionModel
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        del self.study_bias
        del self.qc_linear
        del self.protocol_bias

        hidden_dim = int(kwargs["hidden_dim"])
        self.presence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(kwargs.get("dropout", 0.15))),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_intensity(
        self,
        batch: dict,
        precomputed_unique_local_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.contextual_biology_outputs(
            batch,
            precomputed_unique_local_hidden=precomputed_unique_local_hidden,
        )
        intensity_latent = context["instance_biology_logits"]
        presence_logits = self.presence_head(
            context["instance_context_hidden"]
        ).squeeze(-1)
        return {
            **context,
            "presence_logits": presence_logits,
            "presence_probability": torch.sigmoid(presence_logits),
            "positive_intensity_latent": intensity_latent,
            "positive_intensity_target_space": F.softplus(intensity_latent),
        }


def split_single_head_parameter_groups(
    model: CandidateIntensitySingleHeadHurdleModel,
) -> dict[str, list[nn.Parameter]]:
    intensity_head = list(model.biology_head.parameters())
    presence_head = list(model.presence_head.parameters())
    excluded = {id(p) for p in intensity_head + presence_head}
    backbone = [p for p in model.parameters() if id(p) not in excluded]
    return {
        "backbone": backbone,
        "intensity_head": intensity_head,
        "presence_head": presence_head,
    }
