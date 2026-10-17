#!/usr/bin/env python3
"""Datasets and collators for candidate-level normal m6A MIL training."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


REGION_TO_ID = {"5UTR": 0, "CDS": 1, "3UTR": 2, "other": 3}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


def open_text(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="", encoding="utf-8")


def parse_binary(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return 1
    if text in {"false", "no", "n", ""}:
        return 0
    return int(float(text))


class CandidateFeatureStore:
    """Candidate metadata plus memory-mapped sequence or embedding features."""

    def __init__(
        self,
        candidate_table: str,
        sequence_file: str | None = None,
        embedding_file: str | None = None,
    ) -> None:
        if bool(sequence_file) == bool(embedding_file):
            raise ValueError(
                "Exactly one of sequence_file or embedding_file must be provided"
            )

        candidate_ids: list[str] = []
        site_keys: list[str] = []
        gene_ids: list[str] = []
        transcript_ids: list[str] = []
        chroms: list[str] = []
        strands: list[str] = []
        genomic_positions: list[int] = []
        transcript_positions: list[int] = []
        mature_lengths: list[int] = []
        regions: list[str] = []
        region_ids: list[int] = []
        numeric_features: list[tuple[float, float, float]] = []
        candidate_scores: list[float] = []
        database_support: list[int] = []
        database_sources: list[str] = []
        candidate_tiers: list[str] = []
        split_ids: list[int] = []
        split_names: list[str] = []

        with open_text(candidate_table) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "candidate_index",
                "candidate_id",
                "site_key",
                "gene_id",
                "transcript_id",
                "chrom",
                "strand",
                "genomic_pos0",
                "transcript_pos0",
                "mature_transcript_length",
                "region",
                "region_id",
                "relative_position",
                "is_drach",
                "is_rrach",
                "candidate_score",
                "is_database_site",
                "database_sources",
                "candidate_tier",
                "split",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Candidate metadata is missing columns: {sorted(missing)}"
                )
            for expected_index, row in enumerate(reader):
                actual_index = int(row["candidate_index"])
                if actual_index != expected_index:
                    raise ValueError(
                        "candidate_index must be contiguous and row-aligned: "
                        f"expected {expected_index}, found {actual_index}"
                    )
                candidate_ids.append(row["candidate_id"])
                site_keys.append(row["site_key"])
                gene_ids.append(row["gene_id"])
                transcript_ids.append(row["transcript_id"])
                chroms.append(row["chrom"])
                strands.append(row["strand"])
                genomic_positions.append(int(row["genomic_pos0"]))
                transcript_positions.append(int(row["transcript_pos0"]))
                mature_lengths.append(int(row["mature_transcript_length"]))
                regions.append(row["region"])
                region_ids.append(int(row["region_id"]))
                numeric_features.append(
                    (
                        float(row["relative_position"]),
                        float(parse_binary(row["is_drach"])),
                        float(parse_binary(row["is_rrach"])),
                    )
                )
                candidate_scores.append(float(row["candidate_score"]))
                database_support.append(parse_binary(row["is_database_site"]))
                database_sources.append(row["database_sources"])
                candidate_tiers.append(row["candidate_tier"])
                split_names.append(row["split"])
                split_ids.append(SPLIT_TO_ID.get(row["split"], -1))

        self.candidate_table = str(candidate_table)
        self.candidate_ids = np.asarray(candidate_ids, dtype=object)
        self.site_keys = np.asarray(site_keys, dtype=object)
        self.gene_ids = np.asarray(gene_ids, dtype=object)
        self.transcript_ids = np.asarray(transcript_ids, dtype=object)
        self.chroms = np.asarray(chroms, dtype=object)
        self.strands = np.asarray(strands, dtype=object)
        self.genomic_positions = np.asarray(genomic_positions, dtype=np.int64)
        self.transcript_positions = np.asarray(transcript_positions, dtype=np.int64)
        self.mature_lengths = np.asarray(mature_lengths, dtype=np.int64)
        self.regions = np.asarray(regions, dtype=object)
        self.region_ids = np.asarray(region_ids, dtype=np.int64)
        self.numeric_features = np.asarray(numeric_features, dtype=np.float32)
        self.candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        self.database_support = np.asarray(database_support, dtype=np.int8)
        self.database_sources = np.asarray(database_sources, dtype=object)
        self.candidate_tiers = np.asarray(candidate_tiers, dtype=object)
        self.split_names = np.asarray(split_names, dtype=object)
        self.split_ids = np.asarray(split_ids, dtype=np.int8)

        self.sequence_file = sequence_file
        self.embedding_file = embedding_file
        self.sequences = (
            np.load(sequence_file, mmap_mode="r") if sequence_file else None
        )
        self.embeddings = (
            np.load(embedding_file, mmap_mode="r") if embedding_file else None
        )
        feature_matrix = self.sequences if self.sequences is not None else self.embeddings
        if feature_matrix is None or feature_matrix.shape[0] != len(self.candidate_ids):
            raise ValueError(
                "Feature matrix row count does not match candidate metadata: "
                f"{None if feature_matrix is None else feature_matrix.shape[0]} "
                f"vs {len(self.candidate_ids)}"
            )
        if self.sequences is not None:
            if (
                self.sequences.ndim != 2
                or self.sequences.shape[1] < 3
                or self.sequences.shape[1] % 2 != 1
            ):
                raise ValueError(
                    "Sequence matrix must have shape [N, odd_length>=3], "
                    f"got {self.sequences.shape}"
                )
            if int(self.sequences.min()) < 0 or int(self.sequences.max()) > 4:
                raise ValueError("Sequence tokens must be in the range [0, 4]")
            self.embedding_dim = None
        else:
            if self.embeddings.ndim != 2:
                raise ValueError("Embedding matrix must be two-dimensional")
            self.embedding_dim = int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.candidate_ids)

    @property
    def uses_precomputed_embeddings(self) -> bool:
        return self.embeddings is not None

    def model_inputs(self, candidate_indices: np.ndarray) -> dict[str, torch.Tensor]:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        output = {
            "region_ids": torch.from_numpy(
                self.region_ids[candidate_indices].astype(np.int64, copy=False)
            ),
            "numeric_features": torch.from_numpy(
                self.numeric_features[candidate_indices].astype(np.float32, copy=False)
            ),
        }
        if self.sequences is not None:
            output["sequence_tokens"] = torch.from_numpy(
                np.asarray(self.sequences[candidate_indices], dtype=np.int64)
            )
        else:
            output["precomputed_embeddings"] = torch.from_numpy(
                np.asarray(self.embeddings[candidate_indices], dtype=np.float32)
            )
        return output


class PositiveBagStore:
    """Compact CSR storage of mapped state0 normal peak bags."""

    def __init__(self, mapping_file: str) -> None:
        data = np.load(mapping_file, allow_pickle=False)
        required = {
            "bag_offsets",
            "candidate_indices",
            "study_indices",
            "region_ids",
            "split_ids",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Positive bag mapping is missing arrays: {sorted(missing)}")
        self.bag_offsets = np.asarray(data["bag_offsets"], dtype=np.int64)
        self.candidate_indices = np.asarray(data["candidate_indices"], dtype=np.int64)
        self.study_indices = np.asarray(data["study_indices"], dtype=np.int64)
        self.region_ids = np.asarray(data["region_ids"], dtype=np.int64)
        self.split_ids = np.asarray(data["split_ids"], dtype=np.int64)
        n_bags = len(self.study_indices)
        if len(self.bag_offsets) != n_bags + 1:
            raise ValueError("bag_offsets length must equal n_bags + 1")
        if self.bag_offsets[-1] != len(self.candidate_indices):
            raise ValueError("Final bag offset does not match candidate instance count")
        if len(self.region_ids) != n_bags or len(self.split_ids) != n_bags:
            raise ValueError("Bag-level arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.study_indices)

    def candidates(self, bag_index: int) -> np.ndarray:
        start = self.bag_offsets[bag_index]
        end = self.bag_offsets[bag_index + 1]
        return self.candidate_indices[start:end]


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


@dataclass(frozen=True)
class BagExample:
    candidate_indices: np.ndarray
    study_index: int
    region_id: int
    label: int
    source: str
    source_index: int


class CandidateNormalMILDataset(Dataset):
    """Positive peak bags plus region-matched weak-negative singleton bags."""

    def __init__(
        self,
        feature_store: CandidateFeatureStore,
        bag_store: PositiveBagStore,
        coverage_file: str,
        split: str,
        negative_ratio: int = 3,
        seed: int = 42,
        max_positive_bags: int | None = None,
    ) -> None:
        if split not in SPLIT_TO_ID:
            raise ValueError(f"Unknown split: {split}")
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative")
        self.feature_store = feature_store
        self.bag_store = bag_store
        self.split = split
        self.split_id = SPLIT_TO_ID[split]
        self.negative_ratio = int(negative_ratio)
        self.seed = int(seed)
        self.epoch = 0

        coverage = np.load(coverage_file, mmap_mode="r")
        if coverage.ndim != 2 or coverage.shape[1] != len(feature_store):
            raise ValueError(
                f"Coverage shape {coverage.shape} is incompatible with candidates"
            )
        self.coverage = coverage
        self.n_studies = int(coverage.shape[0])
        if len(bag_store) and int(bag_store.study_indices.max()) >= self.n_studies:
            raise ValueError("Bag study_index exceeds coverage study dimension")

        positive_indices = np.flatnonzero(bag_store.split_ids == self.split_id)
        if max_positive_bags is not None and len(positive_indices) > max_positive_bags:
            rng = np.random.default_rng(seed + self.split_id)
            positive_indices = np.sort(
                rng.choice(
                    positive_indices,
                    size=int(max_positive_bags),
                    replace=False,
                )
            )
        self.positive_bag_indices = positive_indices.astype(np.int64)

        eligible_split = feature_store.split_ids == self.split_id
        self.negative_pools: dict[tuple[int, int], np.ndarray] = {}
        required_pairs = {
            (
                int(bag_store.study_indices[index]),
                int(bag_store.region_ids[index]),
            )
            for index in self.positive_bag_indices
        }
        for study_index, region_id in sorted(required_pairs):
            pool = np.flatnonzero(
                eligible_split
                & (feature_store.region_ids == region_id)
                & (~np.asarray(coverage[study_index], dtype=bool))
            ).astype(np.int64)
            if self.negative_ratio and pool.size == 0:
                raise ValueError(
                    f"No weak negatives for split={split}, study={study_index}, "
                    f"region={region_id}"
                )
            self.negative_pools[(study_index, region_id)] = pool

    @property
    def n_positive_bags(self) -> int:
        return len(self.positive_bag_indices)

    @property
    def n_negative_examples(self) -> int:
        return self.n_positive_bags * self.negative_ratio

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_positive_bags * (1 + self.negative_ratio)

    def __getitem__(self, index: int) -> BagExample:
        group_size = 1 + self.negative_ratio
        positive_local_index = index // group_size
        slot = index % group_size
        positive_bag_index = int(
            self.positive_bag_indices[positive_local_index]
        )
        study_index = int(self.bag_store.study_indices[positive_bag_index])
        region_id = int(self.bag_store.region_ids[positive_bag_index])
        if slot == 0:
            return BagExample(
                candidate_indices=self.bag_store.candidates(positive_bag_index),
                study_index=study_index,
                region_id=region_id,
                label=1,
                source="positive_peak",
                source_index=positive_bag_index,
            )

        pool = self.negative_pools[(study_index, region_id)]
        hash_input = (
            self.seed
            ^ (self.epoch * 0x9E3779B1)
            ^ (positive_bag_index * 0x85EBCA77)
            ^ (slot * 0xC2B2AE3D)
        )
        pool_index = splitmix64(hash_input) % len(pool)
        candidate_index = int(pool[pool_index])
        return BagExample(
            candidate_indices=np.asarray([candidate_index], dtype=np.int64),
            study_index=study_index,
            region_id=region_id,
            label=0,
            source="weak_negative",
            source_index=candidate_index,
        )


class CandidateMILCollator:
    """Deduplicate candidates within a batch before sequence encoding."""

    def __init__(self, feature_store: CandidateFeatureStore) -> None:
        self.feature_store = feature_store

    def __call__(self, examples: list[BagExample]) -> dict:
        if not examples:
            raise ValueError("Cannot collate an empty example list")
        instance_candidate_indices = np.concatenate(
            [example.candidate_indices for example in examples]
        ).astype(np.int64, copy=False)
        unique_candidate_indices, instance_to_unique = np.unique(
            instance_candidate_indices, return_inverse=True
        )
        instance_bag_indices = np.concatenate(
            [
                np.full(len(example.candidate_indices), bag_index, dtype=np.int64)
                for bag_index, example in enumerate(examples)
            ]
        )
        instance_study_indices = np.concatenate(
            [
                np.full(
                    len(example.candidate_indices),
                    example.study_index,
                    dtype=np.int64,
                )
                for example in examples
            ]
        )
        batch = self.feature_store.model_inputs(unique_candidate_indices)
        batch.update(
            {
                "unique_candidate_indices": torch.from_numpy(
                    unique_candidate_indices.astype(np.int64, copy=False)
                ),
                "instance_candidate_indices": torch.from_numpy(
                    instance_candidate_indices
                ),
                "instance_to_unique": torch.from_numpy(
                    instance_to_unique.astype(np.int64, copy=False)
                ),
                "instance_bag_indices": torch.from_numpy(instance_bag_indices),
                "instance_study_indices": torch.from_numpy(instance_study_indices),
                "labels": torch.tensor(
                    [example.label for example in examples], dtype=torch.float32
                ),
                "bag_sizes": torch.tensor(
                    [len(example.candidate_indices) for example in examples],
                    dtype=torch.float32,
                ),
                "bag_study_indices": torch.tensor(
                    [example.study_index for example in examples], dtype=torch.long
                ),
                "bag_region_ids": torch.tensor(
                    [example.region_id for example in examples], dtype=torch.long
                ),
                "sources": [example.source for example in examples],
                "source_indices": torch.tensor(
                    [example.source_index for example in examples], dtype=torch.long
                ),
            }
        )
        return batch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return output
