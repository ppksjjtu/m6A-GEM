#!/usr/bin/env python3
"""Copper-normal-only, condition-balanced, tail-aware Stage1 experiment.

No peak label, bag probability, or MIL objective is used. Every candidate A
receives a zero/nonzero target; positive candidates additionally receive a
continuous log1p-intensity target. Candidate context remains the complete
candidate set of the gene, exactly as in Stage1.

Data contract
-------------
``candidate_table`` and ``sequence_file`` use the existing Stage1
``CandidateFeatureStore`` format.  For the newly selected score>=0.3 universe,
they must be rebuilt row-aligned to that universe; the candidate-screening
model is not used as this trainer's backbone or a supervision target.

``intensity_matrix`` is a NumPy ``.npy`` matrix with shape
``[n_tracks, n_candidates]``.  Values are the already-normalized, non-negative
signal at each candidate coordinate.  Rows must be described by a TSV with:

    track_index    study_index    state_id

``state_id`` is 0 for the paired reference/control state and 1 for its paired
query/stress state.  Optional columns are ``track_id`` and ``track_weight``.
An optional row-aligned ``weight_matrix`` can hold coverage/reliability
weights.  Train/validation/test splitting always comes from the candidate
table's gene-level split, never from individual observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

GENOTYPE_VOCAB = {"<UNK>": 0, "35S:ALKBH10B": 1, "Col-0": 2, "mta": 3}
CONDITION_VOCAB = {
    "<UNK>": 0,
    "unstressed": 1,
    "Pst_DC3000": 2,
    "cold": 3,
    "copper": 4,
    "drought": 5,
    "flg22": 6,
    "salt": 7,
}

from datasets.candidate_m6a_gene_context_dataset import (  # noqa: E402
    AlignedGeneEmbeddingStore,
    CandidateGeneContextMILCollator,
    ContextBatchLayout,
    GeneCandidateIndex,
)
from datasets.candidate_m6a_mil_dataset import (  # noqa: E402
    CandidateFeatureStore,
    SPLIT_TO_ID,
    move_batch_to_device,
)
from models.m6a_candidate_intensity_singlehead_hurdle_model import (  # noqa: E402
    CandidateIntensitySingleHeadHurdleModel,
    split_single_head_parameter_groups,
)
@dataclass(frozen=True)
class TrackSpec:
    track_index: int
    study_index: int
    state_id: int
    track_id: str
    track_weight: float


@dataclass(frozen=True)
class CandidateIntensityGroup:
    candidate_indices: np.ndarray
    intensities: np.ndarray
    weights: np.ndarray
    tail_weights: np.ndarray
    track_index: int
    study_index: int
    state_id: int
    genotype_id: int
    condition_id: int


def normalize_genotype(value: object) -> str:
    text = str(value or "").strip()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    aliases = {
        "": "<UNK>",
        "col0": "Col-0",
        "columbia0": "Col-0",
        "wt": "Col-0",
        "wildtype": "Col-0",
        "mta": "mta",
        "35salkbh10b": "35S:ALKBH10B",
    }
    return aliases.get(compact, text)


def normalize_condition(value: object) -> str:
    text = str(value or "").strip()
    compact = text.lower().replace("-", "_").replace(" ", "_")
    if compact in {"", "control", "ck", "untreated", "unstressed", "hoagland_control"}:
        return "unstressed"
    if "pst" in compact or "dc3000" in compact:
        return "Pst_DC3000"
    return compact


def build_state_mappings(
    studies: list[dict[str, str]],
    genotype_vocab: dict[str, int],
    condition_vocab: dict[str, int],
):
    genotype0, genotype1, condition0, condition1, rows = [], [], [], [], []
    for study in studies:
        genotype_name0 = normalize_genotype(study["state0_genotype"])
        genotype_name1 = normalize_genotype(study["state1_genotype"])
        condition_name0 = normalize_condition(study["state0_condition"])
        condition_name1 = normalize_condition(study["state1_condition"])
        if genotype_name0 not in genotype_vocab or genotype_name1 not in genotype_vocab:
            raise ValueError(f"Genotype vocabulary does not cover {genotype_name0}/{genotype_name1}")
        if condition_name0 not in condition_vocab or condition_name1 not in condition_vocab:
            raise ValueError(f"Condition vocabulary does not cover {condition_name0}/{condition_name1}")
        genotype0.append(genotype_vocab[genotype_name0])
        genotype1.append(genotype_vocab[genotype_name1])
        condition0.append(condition_vocab[condition_name0])
        condition1.append(condition_vocab[condition_name1])
        rows.append({
            "study_index": study["study_index"], "case_id": study["case_id"],
            "state0_genotype": genotype_name0, "state0_genotype_id": genotype0[-1],
            "state0_condition": condition_name0, "state0_condition_id": condition0[-1],
            "state1_genotype": genotype_name1, "state1_genotype_id": genotype1[-1],
            "state1_condition": condition_name1, "state1_condition_id": condition1[-1],
        })
    return (
        np.asarray(genotype0, dtype=np.int64),
        np.asarray(genotype1, dtype=np.int64),
        np.asarray(condition0, dtype=np.int64),
        np.asarray(condition1, dtype=np.int64),
        rows,
    )


def open_text(path: str | Path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return open(path, mode, encoding="utf-8", newline="")


def write_tsv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_studies(path: str) -> list[dict[str, str]]:
    with open_text(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.sort(key=lambda row: int(row["study_index"]))
    for expected, row in enumerate(rows):
        if int(row["study_index"]) != expected:
            raise ValueError("study_index must be contiguous from zero")
    if not rows:
        raise ValueError("study metadata is empty")
    return rows


def parse_state_id(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"0", "state0", "reference", "normal", "control"}:
        return 0
    if text in {"1", "state1", "query", "stress"}:
        return 1
    raise ValueError(f"Unsupported state_id value: {value!r}")


def load_track_specs(path: str, n_tracks: int, n_studies: int) -> list[TrackSpec]:
    with open_text(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"track_index", "study_index", "state_id"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Track metadata is missing columns: {sorted(missing)}")
    specs: list[TrackSpec] = []
    seen: set[int] = set()
    for row in rows:
        track_index = int(row["track_index"])
        study_index = int(row["study_index"])
        if track_index in seen:
            raise ValueError(f"Duplicate track_index: {track_index}")
        if not 0 <= track_index < n_tracks:
            raise ValueError(f"track_index outside matrix: {track_index}")
        if not 0 <= study_index < n_studies:
            raise ValueError(f"study_index outside metadata: {study_index}")
        weight = float(row.get("track_weight", 1.0) or 1.0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid track_weight for track {track_index}")
        specs.append(
            TrackSpec(
                track_index=track_index,
                study_index=study_index,
                state_id=parse_state_id(row["state_id"]),
                track_id=row.get("track_id", f"track_{track_index}"),
                track_weight=weight,
            )
        )
        seen.add(track_index)
    if seen != set(range(n_tracks)):
        missing_rows = sorted(set(range(n_tracks)).difference(seen))[:20]
        raise ValueError(f"Track metadata does not cover matrix rows: {missing_rows}")
    return sorted(specs, key=lambda spec: spec.track_index)


def select_experiment_tracks(
    specs: list[TrackSpec],
    track_condition_ids: np.ndarray,
    *,
    normal_track_id: str,
    normal_condition_id: int,
) -> list[TrackSpec]:
    """Retain one designated Normal plus every genuine stress-state track."""
    normal = [spec for spec in specs if spec.track_id == normal_track_id]
    if len(normal) != 1:
        raise ValueError(
            f"Expected exactly one --normal_track_id match, found {len(normal)}"
        )
    selected = [normal[0]]
    selected.extend(
        spec
        for spec in specs
        if spec.state_id == 1
        and int(track_condition_ids[spec.track_index]) != int(normal_condition_id)
    )
    selected = sorted({spec.track_index: spec for spec in selected}.values(), key=lambda x: x.track_index)
    if len(selected) < 2:
        raise ValueError("No stress tracks remained after experiment filtering")
    return selected


def parse_float_list(text: str, name: str) -> np.ndarray:
    try:
        values = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Invalid {name}: {text!r}") from error
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite comma-separated values")
    return values


def compute_tail_thresholds(
    matrix: np.ndarray,
    specs: list[TrackSpec],
    feature_store: CandidateFeatureStore,
    percentiles: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute positive-intensity percentile cutoffs on training genes only."""
    train_mask = np.asarray(
        feature_store.split_ids == SPLIT_TO_ID["train"], dtype=bool
    )
    thresholds: dict[int, np.ndarray] = {}
    for spec in specs:
        values = np.asarray(matrix[spec.track_index, train_mask], dtype=np.float32)
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size == 0:
            raise ValueError(f"Track {spec.track_id} has no positive training targets")
        thresholds[spec.track_index] = np.quantile(positive, percentiles).astype(
            np.float32
        )
    return thresholds


def audit_intensity_matrix(
    matrix: np.ndarray,
    *,
    allow_missing: bool,
    chunk_size: int = 1_000_000,
) -> dict[str, int | float]:
    finite_count = 0
    missing_count = 0
    zero_count = 0
    minimum = math.inf
    maximum = -math.inf
    flat = matrix.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        values = np.asarray(flat[start : start + chunk_size], dtype=np.float32)
        finite = np.isfinite(values)
        finite_values = values[finite]
        finite_count += int(finite.sum())
        missing_count += int((~finite).sum())
        if finite_values.size:
            local_minimum = float(finite_values.min())
            local_maximum = float(finite_values.max())
            minimum = min(minimum, local_minimum)
            maximum = max(maximum, local_maximum)
            zero_count += int(np.equal(finite_values, 0).sum())
            if local_minimum < 0:
                raise ValueError("Normalized candidate intensities must be non-negative")
    if missing_count and not allow_missing:
        raise ValueError(
            f"Intensity matrix contains {missing_count:,} non-finite values; "
            "pass --allow_missing_intensity only if these should be ignored"
        )
    return {
        "finite_count": finite_count,
        "missing_count": missing_count,
        "zero_count": zero_count,
        "minimum": minimum,
        "maximum": maximum,
    }


class CandidateIntensityMatrixDataset(Dataset):
    """One item is one track x gene group of directly observed candidates."""

    def __init__(
        self,
        feature_store: CandidateFeatureStore,
        gene_index: GeneCandidateIndex,
        intensity_matrix: str,
        track_specs: list[TrackSpec],
        split: str,
        track_genotype_ids: np.ndarray,
        track_condition_ids: np.ndarray,
        tail_thresholds: dict[int, np.ndarray],
        tail_multipliers: np.ndarray,
        weight_matrix: str | None = None,
    ) -> None:
        self.feature_store = feature_store
        self.gene_index = gene_index
        self.intensities = np.load(intensity_matrix, mmap_mode="r")
        self.weights = np.load(weight_matrix, mmap_mode="r") if weight_matrix else None
        self.track_specs = track_specs
        self.track_genotype_ids = np.asarray(track_genotype_ids, dtype=np.int64)
        self.track_condition_ids = np.asarray(track_condition_ids, dtype=np.int64)
        self.tail_thresholds = tail_thresholds
        self.tail_multipliers = np.asarray(tail_multipliers, dtype=np.float32)
        if self.intensities.ndim != 2:
            raise ValueError("intensity_matrix must have shape [tracks, candidates]")
        expected_shape = (self.intensities.shape[0], len(feature_store))
        if self.intensities.shape != expected_shape:
            raise ValueError(
                f"Intensity matrix shape {self.intensities.shape} != {expected_shape}"
            )
        if self.weights is not None and self.weights.shape != expected_shape:
            raise ValueError("weight_matrix must have the same shape as intensity_matrix")
        if split not in SPLIT_TO_ID:
            raise ValueError(f"Unsupported split: {split}")

        split_id = SPLIT_TO_ID[split]
        selected_genes: list[int] = []
        for gene_id in range(len(gene_index)):
            members = gene_index.candidates(gene_id)
            member_splits = np.unique(feature_store.split_ids[members])
            if len(member_splits) != 1:
                name = str(gene_index.gene_ids[gene_id])
                raise ValueError(f"Gene appears in multiple splits: {name}")
            if int(member_splits[0]) == split_id:
                selected_genes.append(gene_id)
        self.gene_indices = np.asarray(selected_genes, dtype=np.int64)
        if not len(self.gene_indices):
            raise ValueError(f"No genes found for split {split}")

    def __len__(self) -> int:
        return len(self.track_specs) * len(self.gene_indices)

    def __getitem__(self, index: int) -> CandidateIntensityGroup:
        track_slot, gene_slot = divmod(index, len(self.gene_indices))
        spec = self.track_specs[track_slot]
        gene_id = int(self.gene_indices[gene_slot])
        candidates = self.gene_index.candidates(gene_id)
        intensities = np.asarray(
            self.intensities[spec.track_index, candidates], dtype=np.float32
        )
        if self.weights is None:
            weights = np.full(len(candidates), spec.track_weight, dtype=np.float32)
        else:
            weights = np.asarray(
                self.weights[spec.track_index, candidates], dtype=np.float32
            ) * np.float32(spec.track_weight)
        valid = np.isfinite(intensities) & np.isfinite(weights) & (weights > 0)
        if np.any(intensities[valid] < 0):
            raise ValueError("Encountered a negative normalized intensity")
        thresholds = self.tail_thresholds[spec.track_index]
        bins = np.searchsorted(thresholds, intensities[valid], side="right")
        tail_weights = self.tail_multipliers[bins]
        return CandidateIntensityGroup(
            candidate_indices=candidates[valid].astype(np.int64, copy=False),
            intensities=intensities[valid],
            weights=weights[valid],
            tail_weights=tail_weights.astype(np.float32, copy=False),
            track_index=spec.track_index,
            study_index=spec.study_index,
            state_id=spec.state_id,
            genotype_id=int(self.track_genotype_ids[spec.track_index]),
            condition_id=int(self.track_condition_ids[spec.track_index]),
        )

    def gene_major_indices(self) -> list[int]:
        """Order evaluation as all tracks of one gene before the next gene.

        This preserves the exact dataset contents while allowing the collator/model
        to de-duplicate the same candidate sequences across conditions in a batch.
        """
        n_genes = len(self.gene_indices)
        return [
            track_slot * n_genes + gene_slot
            for gene_slot in range(n_genes)
            for track_slot in range(len(self.track_specs))
        ]

    def condition_balanced_sampling_weights(self) -> torch.Tensor:
        """Give every retained biological condition equal expected sampling mass."""
        condition_counts: dict[int, int] = defaultdict(int)
        for spec in self.track_specs:
            condition_counts[int(self.track_condition_ids[spec.track_index])] += 1
        per_track = np.asarray(
            [
                1.0 / condition_counts[int(self.track_condition_ids[spec.track_index])]
                for spec in self.track_specs
            ],
            dtype=np.float64,
        )
        return torch.from_numpy(np.repeat(per_track, len(self.gene_indices)))


class CandidateIntensityCollator(CandidateGeneContextMILCollator):
    """Build the unchanged full-gene Stage1 context plus pointwise targets."""

    def __call__(self, examples: list[CandidateIntensityGroup]) -> dict:
        examples = [example for example in examples if len(example.candidate_indices)]
        if not examples:
            raise ValueError("Batch contains no finite candidate intensity")
        candidates = np.concatenate([example.candidate_indices for example in examples])
        studies = np.concatenate(
            [
                np.full(len(example.candidate_indices), example.study_index, np.int64)
                for example in examples
            ]
        )
        genotypes = np.concatenate(
            [
                np.full(len(example.candidate_indices), example.genotype_id, np.int64)
                for example in examples
            ]
        )
        conditions = np.concatenate(
            [
                np.full(len(example.candidate_indices), example.condition_id, np.int64)
                for example in examples
            ]
        )
        batch = self._context_batch(
            ContextBatchLayout(
                instance_candidate_indices=candidates,
                instance_study_indices=studies,
                instance_genotype_ids=genotypes,
                instance_condition_ids=conditions,
            )
        )
        batch.update(
            {
                "intensity_targets": torch.from_numpy(
                    np.concatenate([example.intensities for example in examples])
                ),
                "intensity_weights": torch.from_numpy(
                    np.concatenate([example.weights for example in examples])
                ),
                "tail_weights": torch.from_numpy(
                    np.concatenate([example.tail_weights for example in examples])
                ),
                "target_group_indices": torch.from_numpy(
                    np.concatenate(
                        [
                            np.full(len(example.candidate_indices), group, np.int64)
                            for group, example in enumerate(examples)
                        ]
                    )
                ),
                "instance_track_indices": torch.from_numpy(
                    np.concatenate(
                        [
                            np.full(
                                len(example.candidate_indices),
                                example.track_index,
                                np.int64,
                            )
                            for example in examples
                        ]
                    )
                ),
                "instance_state_ids": torch.from_numpy(
                    np.concatenate(
                        [
                            np.full(
                                len(example.candidate_indices),
                                example.state_id,
                                np.int64,
                            )
                            for example in examples
                        ]
                    )
                ),
            }
        )
        return batch


def make_loader(
    dataset: Dataset,
    collator: CandidateIntensityCollator,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    sampler = None
    if shuffle and args.condition_balanced_sampling:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            dataset.condition_balanced_sampling_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    elif not shuffle and hasattr(dataset, "gene_major_indices"):
        sampler = dataset.gene_major_indices()
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def target_transform(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "log1p":
        return torch.log1p(values)
    if mode == "identity":
        return values
    raise ValueError(f"Unsupported target transform: {mode}")


def invert_target_transform(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "log1p":
        return torch.expm1(values).clamp_min(0)
    if mode == "identity":
        return values.clamp_min(0)
    raise ValueError(f"Unsupported target transform: {mode}")


def sampled_within_group_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    pairs_per_group: int,
    minimum_delta: float,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    zero = predictions.sum() * 0.0
    if pairs_per_group <= 0:
        return zero, 0, zero
    losses: list[torch.Tensor] = []
    pair_weights: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    for group in torch.unique(group_indices):
        indices = torch.nonzero(group_indices.eq(group), as_tuple=False).squeeze(-1)
        n = int(indices.numel())
        if n < 2:
            continue
        draw_count = min(int(pairs_per_group), n * (n - 1) // 2)
        repetitions = max(1, math.ceil(draw_count / n))
        left_parts = []
        right_parts = []
        for _ in range(repetitions):
            left_parts.append(
                indices[torch.randperm(n, device=indices.device, generator=generator)]
            )
            right_parts.append(
                indices[torch.randperm(n, device=indices.device, generator=generator)]
            )
        left = torch.cat(left_parts)[:draw_count]
        right = torch.cat(right_parts)[:draw_count]
        valid = left.ne(right)
        target_delta = targets[left] - targets[right]
        valid &= target_delta.abs().ge(float(minimum_delta))
        if not valid.any():
            continue
        left = left[valid]
        right = right[valid]
        target_delta = target_delta[valid]
        direction = target_delta.sign()
        prediction_delta = predictions[left] - predictions[right]
        pair_weight = torch.sqrt(weights[left] * weights[right]).clamp_min(0)
        pair_loss = F.softplus(
            -direction * prediction_delta / float(temperature)
        ) * float(temperature)
        losses.append(pair_loss)
        pair_weights.append(pair_weight)
        accuracies.append((direction * prediction_delta > 0).float())
    if not losses:
        return zero, 0, zero
    losses_tensor = torch.cat(losses)
    weights_tensor = torch.cat(pair_weights)
    accuracy_tensor = torch.cat(accuracies)
    denominator = weights_tensor.sum().clamp_min(torch.finfo(weights_tensor.dtype).eps)
    loss = (losses_tensor * weights_tensor).sum() / denominator
    accuracy = (accuracy_tensor * weights_tensor).sum() / denominator
    return loss, int(losses_tensor.numel()), accuracy


def training_objective(
    model: CandidateIntensitySingleHeadHurdleModel,
    batch: dict,
    args: argparse.Namespace,
    *,
    ranking_generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor | int]:
    output = model.forward_intensity(batch)
    targets = target_transform(batch["intensity_targets"], args.target_transform)
    positive_targets = batch["intensity_targets"].gt(args.zero_epsilon).float()
    positive_mask = positive_targets.bool()
    predictions = output["positive_intensity_target_space"]
    weights = batch["intensity_weights"].float().clamp_min(0)
    tail_weights = batch["tail_weights"].float().clamp_min(0)
    presence_pointwise = F.binary_cross_entropy_with_logits(
        output["presence_logits"].float(), positive_targets.float(),
        pos_weight=torch.as_tensor(
            args.presence_positive_weight,
            dtype=torch.float32,
            device=positive_targets.device,
        ),
        reduction="none",
    )
    presence_loss = (
        presence_pointwise * weights
    ).sum() / weights.sum().clamp_min(1e-12)
    pointwise = F.huber_loss(
        predictions.float(),
        targets.float(),
        delta=args.huber_delta,
        reduction="none",
    )
    positive_weights = weights * tail_weights * positive_mask.float()
    regression = (
        pointwise * positive_weights
    ).sum() / positive_weights.sum().clamp_min(1e-12)
    ranking, pair_count, pair_accuracy = sampled_within_group_ranking_loss(
        predictions.float()[positive_mask],
        targets.float()[positive_mask],
        batch["target_group_indices"][positive_mask],
        (weights * tail_weights)[positive_mask],
        pairs_per_group=args.ranking_pairs_per_group,
        minimum_delta=args.ranking_minimum_delta,
        temperature=args.ranking_temperature,
        generator=ranking_generator,
    )
    loss = (
        args.presence_loss_weight * presence_loss
        + regression
        + args.ranking_weight * ranking
    )
    positive_original = invert_target_transform(
        predictions.float(), args.target_transform
    )
    expected_original = output["presence_probability"].float() * positive_original
    expected_target_space = target_transform(
        expected_original, args.target_transform
    )
    sparse_mask = output["presence_probability"].ge(args.presence_threshold)
    sparse_original = torch.where(
        sparse_mask, positive_original, torch.zeros_like(positive_original)
    )
    sparse_target_space = target_transform(
        sparse_original, args.target_transform
    )
    return {
        **output,
        "loss": loss,
        "presence_loss": presence_loss,
        "regression_loss": regression,
        "ranking_loss": ranking,
        "ranking_pair_count": pair_count,
        "ranking_accuracy": pair_accuracy,
        "targets_target_space": targets,
        "positive_predictions_original": positive_original,
        "expected_predictions_target_space": expected_target_space,
        "expected_predictions_original": expected_original,
        "sparse_predictions_target_space": sparse_target_space,
        "sparse_predictions_original": sparse_original,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


class RegressionAccumulator:
    def __init__(self, maximum_rank_points: int, seed: int) -> None:
        self.maximum_rank_points = int(maximum_rank_points)
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.target_sum = 0.0
        self.prediction_sum = 0.0
        self.target_squared_sum = 0.0
        self.prediction_squared_sum = 0.0
        self.cross_sum = 0.0
        self.rank_targets: list[np.ndarray] = []
        self.rank_predictions: list[np.ndarray] = []
        self.rank_count = 0

    def update(self, targets: np.ndarray, predictions: np.ndarray, quota: int) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        residual = predictions - targets
        self.n += len(targets)
        self.absolute_error_sum += float(np.abs(residual).sum())
        self.squared_error_sum += float(np.square(residual).sum())
        self.target_sum += float(targets.sum())
        self.prediction_sum += float(predictions.sum())
        self.target_squared_sum += float(np.square(targets).sum())
        self.prediction_squared_sum += float(np.square(predictions).sum())
        self.cross_sum += float((targets * predictions).sum())
        remaining = self.maximum_rank_points - self.rank_count
        take = min(len(targets), max(0, min(quota, remaining)))
        if take:
            selected = self.rng.choice(len(targets), size=take, replace=False)
            self.rank_targets.append(targets[selected].astype(np.float32))
            self.rank_predictions.append(predictions[selected].astype(np.float32))
            self.rank_count += take

    def finalize(self) -> dict[str, int | float]:
        if not self.n:
            return {
                "n": 0,
                "mae": math.nan,
                "rmse": math.nan,
                "pearson": math.nan,
                "spearman": math.nan,
                "spearman_n": 0,
            }
        target_variance = self.target_squared_sum - self.target_sum**2 / self.n
        prediction_variance = (
            self.prediction_squared_sum - self.prediction_sum**2 / self.n
        )
        covariance = self.cross_sum - self.target_sum * self.prediction_sum / self.n
        denominator = math.sqrt(max(target_variance * prediction_variance, 0.0))
        pearson = covariance / denominator if denominator > 0 else math.nan
        spearman = math.nan
        if self.rank_count >= 2:
            targets = np.concatenate(self.rank_targets)
            predictions = np.concatenate(self.rank_predictions)
            target_ranks = average_ranks(targets)
            prediction_ranks = average_ranks(predictions)
            spearman = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
        return {
            "n": self.n,
            "mae": self.absolute_error_sum / self.n,
            "rmse": math.sqrt(self.squared_error_sum / self.n),
            "pearson": pearson,
            "spearman": spearman,
            "spearman_n": self.rank_count,
        }


def train_epoch(
    model: CandidateIntensitySingleHeadHurdleModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    model.train()
    totals = defaultdict(float)
    candidate_count = 0
    pair_count = 0
    optimizer_steps = 0
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=args.amp and device.type == "cuda",
        ):
            output = training_objective(model, batch, args)
        loss = output["loss"]
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / args.gradient_accumulation_steps).backward()
        should_step = (
            (batch_index + 1) % args.gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            )
            if torch.isfinite(gradient_norm):
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
            else:
                skipped += 1
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
        count = int(batch["intensity_targets"].numel())
        candidate_count += count
        pair_count += int(output["ranking_pair_count"])
        for key in (
            "loss",
            "presence_loss",
            "regression_loss",
            "ranking_loss",
            "ranking_accuracy",
        ):
            totals[key] += float(output[key].detach().cpu()) * count
        if args.log_every_n_batches and (batch_index + 1) % args.log_every_n_batches == 0:
            print(
                f"[train {batch_index + 1:,}/{len(loader):,}] "
                f"loss={totals['loss'] / max(candidate_count, 1):.6f} "
                f"candidates={candidate_count:,} steps={optimizer_steps:,}",
                flush=True,
            )
    denominator = max(candidate_count, 1)
    return {
        **{key: value / denominator for key, value in totals.items()},
        "candidate_count": candidate_count,
        "ranking_pair_count": pair_count,
        "optimizer_steps": optimizer_steps,
        "skipped_batches_or_steps": skipped,
    }


def export_batch_predictions(
    writer: csv.DictWriter,
    output: dict,
    batch: dict,
    feature_store: CandidateFeatureStore,
    track_specs: list[TrackSpec],
    genotype_names: dict[int, str],
    condition_names: dict[int, str],
) -> None:
    tensor_names = (
        "instance_candidate_indices",
        "instance_study_indices",
        "instance_genotype_ids",
        "instance_condition_ids",
        "instance_track_indices",
        "instance_state_ids",
        "intensity_targets",
        "tail_weights",
    )
    values = {name: batch[name].detach().cpu().numpy() for name in tensor_names}
    prediction = output["expected_predictions_original"].detach().float().cpu().numpy()
    prediction_log1p = output["expected_predictions_target_space"].detach().float().cpu().numpy()
    sparse_prediction = output["sparse_predictions_original"].detach().float().cpu().numpy()
    sparse_prediction_log1p = output["sparse_predictions_target_space"].detach().float().cpu().numpy()
    positive_prediction = output["positive_predictions_original"].detach().float().cpu().numpy()
    positive_prediction_log1p = output["positive_intensity_target_space"].detach().float().cpu().numpy()
    intensity_latent = output["positive_intensity_latent"].detach().float().cpu().numpy()
    presence_probability = output["presence_probability"].detach().float().cpu().numpy()
    track_spec_by_index = {spec.track_index: spec for spec in track_specs}
    for row_index, raw_candidate_index in enumerate(values["instance_candidate_indices"]):
        candidate_index = int(raw_candidate_index)
        track_index = int(values["instance_track_indices"][row_index])
        genotype_id = int(values["instance_genotype_ids"][row_index])
        condition_id = int(values["instance_condition_ids"][row_index])
        writer.writerow(
            {
                "candidate_index": candidate_index,
                "candidate_id": feature_store.candidate_ids[candidate_index],
                "gene_id": feature_store.gene_ids[candidate_index],
                "transcript_id": feature_store.transcript_ids[candidate_index],
                "track_index": track_index,
                "track_id": track_spec_by_index[track_index].track_id,
                "study_index": int(values["instance_study_indices"][row_index]),
                "state_id": int(values["instance_state_ids"][row_index]),
                "genotype": genotype_names[genotype_id],
                "condition": condition_names[condition_id],
                "normalized_intensity": float(values["intensity_targets"][row_index]),
                "normalized_log1p_intensity": float(np.log1p(values["intensity_targets"][row_index])),
                "observed_is_positive": int(values["intensity_targets"][row_index] > 0),
                "tail_weight": float(values["tail_weights"][row_index]),
                "presence_probability": float(presence_probability[row_index]),
                "predicted_intensity": float(prediction[row_index]),
                "predicted_log1p_intensity": float(prediction_log1p[row_index]),
                "sparse_predicted_intensity": float(sparse_prediction[row_index]),
                "sparse_predicted_log1p_intensity": float(sparse_prediction_log1p[row_index]),
                "positive_predicted_intensity": float(positive_prediction[row_index]),
                "positive_predicted_log1p_intensity": float(positive_prediction_log1p[row_index]),
                "positive_intensity_latent": float(intensity_latent[row_index]),
            }
        )


@torch.no_grad()
def evaluate(
    model: CandidateIntensitySingleHeadHurdleModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    seed_offset: int,
    prediction_writer: csv.DictWriter | None = None,
    feature_store: CandidateFeatureStore | None = None,
    track_specs: list[TrackSpec] | None = None,
    genotype_names: dict[int, str] | None = None,
    condition_names: dict[int, str] | None = None,
) -> dict[str, float | int]:
    model.eval()
    accumulator = RegressionAccumulator(
        args.metrics_max_rank_points, args.seed + seed_offset
    )
    sparse_accumulator = RegressionAccumulator(
        args.metrics_max_rank_points, args.seed + seed_offset + 1
    )
    totals = defaultdict(float)
    candidate_count = 0
    pair_count = 0
    pair_correct_weighted = 0.0
    presence_correct = 0
    presence_tp = 0
    presence_fp = 0
    presence_fn = 0
    rank_quota = max(1, math.ceil(args.metrics_max_rank_points / max(len(loader), 1)))
    ranking_generator = torch.Generator(device=device)
    ranking_generator.manual_seed(args.seed + seed_offset)
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=args.amp and device.type == "cuda",
        ):
            output = training_objective(
                model, batch, args, ranking_generator=ranking_generator
            )
        count = int(batch["intensity_targets"].numel())
        candidates = batch["intensity_targets"].detach().cpu().numpy()
        predictions = output["expected_predictions_original"].detach().float().cpu().numpy()
        sparse_predictions = output["sparse_predictions_original"].detach().float().cpu().numpy()
        accumulator.update(candidates, predictions, rank_quota)
        sparse_accumulator.update(candidates, sparse_predictions, rank_quota)
        truth_positive = candidates > args.zero_epsilon
        called_positive = (
            output["presence_probability"].detach().float().cpu().numpy()
            >= args.presence_threshold
        )
        presence_correct += int(np.equal(truth_positive, called_positive).sum())
        presence_tp += int((truth_positive & called_positive).sum())
        presence_fp += int((~truth_positive & called_positive).sum())
        presence_fn += int((truth_positive & ~called_positive).sum())
        candidate_count += count
        current_pairs = int(output["ranking_pair_count"])
        pair_count += current_pairs
        pair_correct_weighted += float(output["ranking_accuracy"].detach().cpu()) * current_pairs
        for key in ("loss", "presence_loss", "regression_loss", "ranking_loss"):
            totals[key] += float(output[key].detach().cpu()) * count
        if prediction_writer is not None:
            if any(
                value is None
                for value in (
                    feature_store,
                    track_specs,
                    genotype_names,
                    condition_names,
                )
            ):
                raise ValueError("Prediction export metadata was not provided")
            export_batch_predictions(
                prediction_writer,
                output,
                batch,
                feature_store,
                track_specs,
                genotype_names,
                condition_names,
            )
    denominator = max(candidate_count, 1)
    precision = presence_tp / max(presence_tp + presence_fp, 1)
    recall = presence_tp / max(presence_tp + presence_fn, 1)
    sparse_metrics = {
        f"sparse_{key}": value
        for key, value in sparse_accumulator.finalize().items()
    }
    return {
        **{key: value / denominator for key, value in totals.items()},
        "ranking_pair_count": pair_count,
        "ranking_accuracy": pair_correct_weighted / max(pair_count, 1),
        "presence_accuracy": presence_correct / denominator,
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        **accumulator.finalize(),
        **sparse_metrics,
    }


def build_model_from_scratch(
    feature_store: CandidateFeatureStore,
    gene_store: AlignedGeneEmbeddingStore,
    qc_features: torch.Tensor,
    protocol_ids: torch.Tensor,
    genotype_vocab: dict[str, int],
    condition_vocab: dict[str, int],
    args: argparse.Namespace,
) -> CandidateIntensitySingleHeadHurdleModel:
    return CandidateIntensitySingleHeadHurdleModel(
        num_studies=int(qc_features.shape[0]),
        num_genotypes=len(genotype_vocab),
        num_conditions=len(condition_vocab),
        condition_reference_id=condition_vocab["unstressed"],
        num_protocols=int(protocol_ids.max()) + 1,
        qc_features=qc_features,
        protocol_ids=protocol_ids,
        gene_embedding_dim=gene_store.embedding_dim,
        sequence_length=int(feature_store.sequences.shape[1]),
        sequence_encoder_dim=args.sequence_encoder_dim,
        sequence_encoder_layers=args.sequence_encoder_layers,
        sequence_encoder_kernel_size=7,
        sequence_encoder_dropout=args.dropout,
        sequence_chunk_size=args.sequence_chunk_size,
        sequence_transformer_heads=args.sequence_transformer_heads,
        sequence_transformer_ffn_multiplier=args.sequence_transformer_ffn_multiplier,
        sequence_gradient_checkpointing=True,
        num_regions=int(feature_store.region_ids.max()) + 1,
        region_embedding_dim=args.region_embedding_dim,
        numeric_feature_dim=int(feature_store.numeric_features.shape[1]),
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_multiplier=args.transformer_ffn_multiplier,
        gene_token_count=args.gene_token_count,
        dropout=args.dropout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="AgroNT Stage1 candidate-level continuous intensity regression",
    )
    parser.add_argument("--candidate_table", required=True)
    parser.add_argument("--sequence_file", required=True)
    parser.add_argument("--intensity_matrix", required=True)
    parser.add_argument("--intensity_track_metadata", required=True)
    parser.add_argument("--weight_matrix")
    parser.add_argument("--study_metadata", required=True)
    parser.add_argument("--agront_embedding_file", required=True)
    parser.add_argument("--agront_gene_ids_file", required=True)
    parser.add_argument("--agront_availability_file", required=True)
    parser.add_argument("--qc_inputs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--normal_track_id",
        default="Sharma_2024_copper_control_excess_cpm",
        help="The only Normal/control track retained in this experiment.",
    )
    parser.add_argument("--resume_checkpoint")
    parser.add_argument(
        "--inference_checkpoint",
        help=(
            "Strictly load model weights for --eval_only_split without restoring "
            "optimizer/training state. Required in eval-only mode."
        ),
    )
    parser.add_argument(
        "--init_checkpoint",
        help="Optional compatible scratch401 checkpoint used to initialize weights.",
    )
    parser.add_argument("--allow_missing_intensity", action="store_true")

    parser.add_argument("--sequence_encoder_dim", type=int, default=192)
    parser.add_argument("--sequence_encoder_layers", type=int, default=6)
    parser.add_argument("--sequence_chunk_size", type=int, default=32)
    parser.add_argument("--sequence_transformer_heads", type=int, default=8)
    parser.add_argument("--sequence_transformer_ffn_multiplier", type=int, default=4)
    parser.add_argument("--region_embedding_dim", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_ffn_multiplier", type=int, default=4)
    parser.add_argument("--gene_token_count", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--target_transform", choices=("log1p", "identity"), default="log1p")
    parser.add_argument("--huber_delta", type=float, default=0.5)
    parser.add_argument("--zero_epsilon", type=float, default=0.0)
    parser.add_argument("--presence_loss_weight", type=float, default=1.0)
    parser.add_argument("--presence_positive_weight", type=float, default=1.0)
    parser.add_argument("--presence_threshold", type=float, default=0.5)
    parser.add_argument("--ranking_weight", type=float, default=0.10)
    parser.add_argument("--ranking_pairs_per_group", type=int, default=256)
    parser.add_argument("--ranking_minimum_delta", type=float, default=0.05)
    parser.add_argument("--ranking_temperature", type=float, default=0.20)
    parser.add_argument(
        "--tail_percentiles", default="0.50,0.80,0.90,0.95,0.99"
    )
    parser.add_argument(
        "--tail_multipliers", default="1.0,1.5,2.0,3.0,4.0,5.0"
    )
    parser.add_argument(
        "--condition_balanced_sampling", action="store_true", default=True
    )
    parser.add_argument(
        "--no_condition_balanced_sampling",
        action="store_false",
        dest="condition_balanced_sampling",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--backbone_learning_rate", type=float, default=1e-5)
    parser.add_argument("--intensity_head_learning_rate", type=float, default=3e-5)
    parser.add_argument("--presence_head_learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--log_every_n_batches", type=int, default=100)
    parser.add_argument("--metrics_max_rank_points", type=int, default=1_000_000)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_prediction_exports", action="store_true")
    parser.add_argument("--eval_only_split", choices=("train", "val", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only_split and not args.inference_checkpoint:
        raise ValueError("--eval_only_split requires --inference_checkpoint")
    if args.inference_checkpoint and not args.eval_only_split:
        raise ValueError("--inference_checkpoint is only valid with --eval_only_split")
    if args.inference_checkpoint and (args.resume_checkpoint or args.init_checkpoint):
        raise ValueError(
            "Inference-only loading cannot be combined with resume/init checkpoints"
        )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.ranking_temperature <= 0:
        raise ValueError("ranking_temperature must be positive")
    tail_percentiles = parse_float_list(args.tail_percentiles, "tail_percentiles")
    tail_multipliers = parse_float_list(args.tail_multipliers, "tail_multipliers")
    if not np.all((tail_percentiles > 0) & (tail_percentiles < 1)):
        raise ValueError("tail_percentiles must lie strictly between zero and one")
    if not np.all(np.diff(tail_percentiles) > 0):
        raise ValueError("tail_percentiles must be strictly increasing")
    if len(tail_multipliers) != len(tail_percentiles) + 1:
        raise ValueError("tail_multipliers must have len(tail_percentiles) + 1 values")
    if np.any(tail_multipliers <= 0):
        raise ValueError("tail_multipliers must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_store = CandidateFeatureStore(
        args.candidate_table, sequence_file=args.sequence_file
    )
    gene_index = GeneCandidateIndex(feature_store)
    gene_store = AlignedGeneEmbeddingStore(
        args.agront_embedding_file,
        args.agront_gene_ids_file,
        args.agront_availability_file,
        gene_index,
    )
    intensities = np.load(args.intensity_matrix, mmap_mode="r")
    if intensities.ndim != 2 or intensities.shape[1] != len(feature_store):
        raise ValueError(
            "intensity_matrix must be [tracks, candidates] and row-aligned "
            "to candidate_table"
        )
    studies = load_studies(args.study_metadata)
    all_track_specs = load_track_specs(
        args.intensity_track_metadata, intensities.shape[0], len(studies)
    )
    genotype_vocab = dict(GENOTYPE_VOCAB)
    condition_vocab = dict(CONDITION_VOCAB)
    genotype0, genotype1, condition0, condition1, mapping_rows = build_state_mappings(
        studies, genotype_vocab, condition_vocab
    )
    track_genotypes = np.asarray(
        [
            genotype0[spec.study_index] if spec.state_id == 0 else genotype1[spec.study_index]
            for spec in all_track_specs
        ],
        dtype=np.int64,
    )
    track_conditions = np.asarray(
        [
            condition0[spec.study_index] if spec.state_id == 0 else condition1[spec.study_index]
            for spec in all_track_specs
        ],
        dtype=np.int64,
    )
    track_specs = select_experiment_tracks(
        all_track_specs,
        track_conditions,
        normal_track_id=args.normal_track_id,
        normal_condition_id=condition_vocab["unstressed"],
    )
    selected_track_indices = np.asarray(
        [spec.track_index for spec in track_specs], dtype=np.int64
    )
    matrix_audit = audit_intensity_matrix(
        intensities[selected_track_indices],
        allow_missing=args.allow_missing_intensity,
    )
    tail_thresholds = compute_tail_thresholds(
        intensities, track_specs, feature_store, tail_percentiles
    )
    qc = np.load(args.qc_inputs, allow_pickle=False)
    qc_features = torch.from_numpy(qc["qc_features"])
    protocol_ids = torch.from_numpy(qc["protocol_ids"])
    if qc_features.shape[0] != len(studies) or protocol_ids.shape != (len(studies),):
        raise ValueError("QC inputs must contain one row per study")

    model = build_model_from_scratch(
        feature_store,
        gene_store,
        qc_features,
        protocol_ids,
        genotype_vocab,
        condition_vocab,
        args,
    ).to(device)
    if args.inference_checkpoint:
        inference_payload = torch.load(
            args.inference_checkpoint, map_location="cpu", weights_only=False
        )
        inference_state = inference_payload.get("model_state_dict")
        if inference_state is None:
            raise KeyError("inference_checkpoint has no model_state_dict")
        model.load_state_dict(inference_state, strict=True)
    if args.init_checkpoint:
        init_payload = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        init_state = init_payload.get("model_state_dict")
        if init_state is None:
            raise KeyError("init_checkpoint has no model_state_dict")
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        allowed_missing = {
            "presence_head.0.weight", "presence_head.0.bias",
            "presence_head.1.weight", "presence_head.1.bias",
            "presence_head.4.weight", "presence_head.4.bias",
        }
        old_head2 = {
            "case_raw_scale", "case_bias",
            "case_state_raw_scale", "case_state_bias",
        }
        if set(missing).difference(allowed_missing) or set(unexpected).difference(old_head2):
            raise RuntimeError(
                f"Incompatible init checkpoint; missing={missing}, unexpected={unexpected}"
            )
    parameter_groups = split_single_head_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": parameter_groups["backbone"],
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
                "name": "backbone",
            },
            {
                "params": parameter_groups["intensity_head"],
                "lr": args.intensity_head_learning_rate,
                "weight_decay": args.weight_decay,
                "name": "intensity_head",
            },
            {
                "params": parameter_groups["presence_head"],
                "lr": args.presence_head_learning_rate,
                "weight_decay": args.weight_decay,
                "name": "presence_head",
            },
        ]
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and args.amp_dtype == "float16" and device.type == "cuda",
    )

    datasets = {
        split: CandidateIntensityMatrixDataset(
            feature_store,
            gene_index,
            args.intensity_matrix,
            track_specs,
            split,
            track_genotypes,
            track_conditions,
            tail_thresholds,
            tail_multipliers,
            weight_matrix=args.weight_matrix,
        )
        for split in ("train", "val", "test")
    }
    collator = CandidateIntensityCollator(
        feature_store,
        gene_index,
        np.zeros(len(studies), dtype=np.int64),
        len(genotype_vocab),
        gene_store,
        num_conditions=len(condition_vocab),
    )
    loaders = {
        split: make_loader(
            dataset, collator, args, shuffle=split == "train"
        )
        for split, dataset in datasets.items()
    }

    config = {
        **vars(args),
        "model_version": "agront-scratch401-copper-normal-singlehead-tailbalanced-v1",
        "initialization": (
            "scratch401_checkpoint" if args.init_checkpoint
            else "all model parameters random; no Stage1 checkpoint"
        ),
        "sequence_encoder_architecture": "pure_transformer",
        "sequence_length": int(feature_store.sequences.shape[1]),
        "supervision": "presence_BCE_plus_tail_aware_positive_log1p_intensity",
        "peak_labels_used": False,
        "mil_used": False,
        "presence_head_used": True,
        "window_head_used": False,
        "intensity_head": "single_direct_positive_intensity_head",
        "observation_head": None,
        "primary_prediction": "presence_probability_times_positive_intensity",
        "secondary_prediction": "thresholded_sparse_intensity",
        "normal_track_id": args.normal_track_id,
        "condition_balanced_sampling": args.condition_balanced_sampling,
        "tail_percentiles_parsed": tail_percentiles.tolist(),
        "tail_multipliers_parsed": tail_multipliers.tolist(),
        "tail_thresholds_by_track": {
            str(key): value.tolist() for key, value in tail_thresholds.items()
        },
        "ranking_scope": "within_track_state_gene_positive_targets_only",
        "genotype_vocab": genotype_vocab,
        "condition_vocab": condition_vocab,
        "matrix_audit": matrix_audit,
        "n_candidates": len(feature_store),
        "n_genes": len(gene_index),
        "n_tracks": len(track_specs),
        "selected_track_indices": selected_track_indices.tolist(),
        "selected_track_ids": [spec.track_id for spec in track_specs],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_groups": {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in parameter_groups.items()
        },
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    write_tsv(output_dir / "paired_condition_mapping.tsv", mapping_rows)
    write_tsv(
        output_dir / "intensity_track_metadata.resolved.tsv",
        [
            {
                "track_index": spec.track_index,
                "track_id": spec.track_id,
                "study_index": spec.study_index,
                "state_id": spec.state_id,
                "genotype": next(k for k, v in genotype_vocab.items() if v == track_genotypes[spec.track_index]),
                "condition": next(k for k, v in condition_vocab.items() if v == track_conditions[spec.track_index]),
                "track_weight": spec.track_weight,
            }
            for spec in track_specs
        ],
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": config["parameter_count"],
                "candidates": len(feature_store),
                "genes": len(gene_index),
                "tracks": len(track_specs),
                "dataset_groups": {key: len(value) for key, value in datasets.items()},
                "AgroNT": f"{gene_store.embedding_dim} -> {model.gene_token_count}x{model.hidden_dim}",
            },
            indent=2,
        ),
        flush=True,
    )

    start_epoch = 1
    best_metric = -math.inf
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    elapsed_offset = 0.0
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        if resume.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_metric = float(resume.get("best_metric", -math.inf))
        best_epoch = int(resume.get("best_epoch", 0))
        patience = int(resume.get("patience", 0))
        metrics_path = output_dir / "epoch_metrics.tsv"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle, delimiter="\t"))
            if history and int(history[-1]["epoch"]) != start_epoch - 1:
                raise ValueError(
                    "Existing epoch_metrics.tsv ends at epoch "
                    f"{history[-1]['epoch']}, but resume checkpoint starts at "
                    f"epoch {start_epoch}"
                )
            if history:
                elapsed_offset = float(history[-1]["elapsed_seconds"])

    if args.eval_only_split:
        split = args.eval_only_split
        genotype_names = {value: key for key, value in genotype_vocab.items()}
        condition_names = {value: key for key, value in condition_vocab.items()}
        fields = [
            "candidate_index", "candidate_id", "gene_id", "transcript_id",
            "track_index", "track_id", "study_index", "state_id", "genotype",
            "condition", "normalized_intensity", "normalized_log1p_intensity",
            "observed_is_positive", "tail_weight", "presence_probability",
            "predicted_intensity", "predicted_log1p_intensity",
            "sparse_predicted_intensity", "sparse_predicted_log1p_intensity",
            "positive_predicted_intensity", "positive_predicted_log1p_intensity",
            "positive_intensity_latent",
        ]
        prediction_path = output_dir / f"{split}_candidate_intensity_predictions.tsv.gz"
        with gzip.open(prediction_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            metrics = evaluate(
                model,
                loaders[split],
                device,
                args,
                seed_offset=30_000,
                prediction_writer=writer,
                feature_store=feature_store,
                track_specs=track_specs,
                genotype_names=genotype_names,
                condition_names=condition_names,
            )
        with (output_dir / f"eval_only_{split}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"split": split, "metrics": metrics, "predictions": str(prediction_path)}, indent=2))
        return

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        training = train_epoch(model, loaders["train"], optimizer, scaler, device, args)
        validation = evaluate(
            model, loaders["val"], device, args, seed_offset=10_000
        )
        # Select a checkpoint on the complete hurdle objective, not ranking alone.
        metric = -float(validation["loss"])
        if math.isfinite(metric) and metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "elapsed_seconds": elapsed_offset + time.time() - start_time,
            **{f"train_{key}": value for key, value in training.items()},
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        write_tsv(output_dir / "epoch_metrics.tsv", history)
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "patience": patience,
            "args": vars(args),
            "model_version": config["model_version"],
            "genotype_vocab": genotype_vocab,
            "condition_vocab": condition_vocab,
        }
        torch.save(payload, output_dir / "last_model.pt")
        if best_epoch == epoch:
            torch.save(payload, output_dir / "best_model.pt")
        print(
            f"[epoch {epoch:03d}] train_loss={training['loss']:.6f} "
            f"val_RMSE={validation['rmse']:.6f} "
            f"val_loss={validation['loss']:.6f} val_Spearman={validation['spearman']:.6f} "
            f"val_presence_F1={validation['presence_f1']:.6f} best_score={best_metric:.6f}@{best_epoch} "
            f"patience={patience}/{args.early_stopping_patience}",
            flush=True,
        )
        if patience >= args.early_stopping_patience:
            break

    best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    final_metrics = {
        split: evaluate(
            model, loaders[split], device, args, seed_offset=20_000 + index
        )
        for index, split in enumerate(("val", "test"))
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "checkpoint_selection": "minimum validation hurdle loss",
                "best_validation_score_negative_loss": best_metric,
                "metrics": final_metrics,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    if not args.skip_prediction_exports:
        genotype_names = {value: key for key, value in genotype_vocab.items()}
        condition_names = {value: key for key, value in condition_vocab.items()}
        fields = [
            "candidate_index",
            "candidate_id",
            "gene_id",
            "transcript_id",
            "track_index",
            "track_id",
            "study_index",
            "state_id",
            "genotype",
            "condition",
            "normalized_intensity",
            "normalized_log1p_intensity",
            "observed_is_positive",
            "tail_weight",
            "presence_probability",
            "predicted_intensity",
            "predicted_log1p_intensity",
            "sparse_predicted_intensity",
            "sparse_predicted_log1p_intensity",
            "positive_predicted_intensity",
            "positive_predicted_log1p_intensity",
            "positive_intensity_latent",
        ]
        for split in ("val", "test"):
            with gzip.open(
                output_dir / f"{split}_candidate_intensity_predictions.tsv.gz",
                "wt",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                evaluate(
                    model,
                    loaders[split],
                    device,
                    args,
                    seed_offset=30_000,
                    prediction_writer=writer,
                    feature_store=feature_store,
                    track_specs=track_specs,
                    genotype_names=genotype_names,
                    condition_names=condition_names,
                )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "checkpoint_selection": "minimum validation hurdle loss",
                "best_validation_score_negative_loss": best_metric,
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
