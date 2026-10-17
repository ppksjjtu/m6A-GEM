#!/usr/bin/env python3
"""Gene-level candidate context batching for normal m6A MIL models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from datasets.candidate_m6a_mil_dataset import BagExample, CandidateFeatureStore


class GeneCandidateIndex:
    """CSR index of every candidate A belonging to each gene."""

    def __init__(self, feature_store: CandidateFeatureStore) -> None:
        gene_to_index: dict[str, int] = {}
        gene_ids: list[str] = []
        members: list[list[int]] = []
        candidate_gene_indices = np.empty(len(feature_store), dtype=np.int64)

        for candidate_index, raw_gene_id in enumerate(feature_store.gene_ids):
            gene_id = str(raw_gene_id)
            gene_index = gene_to_index.get(gene_id)
            if gene_index is None:
                gene_index = len(gene_ids)
                gene_to_index[gene_id] = gene_index
                gene_ids.append(gene_id)
                members.append([])
            members[gene_index].append(candidate_index)
            candidate_gene_indices[candidate_index] = gene_index

        offsets = np.zeros(len(gene_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(values) for values in members])
        candidate_indices = np.concatenate(
            [np.asarray(values, dtype=np.int64) for values in members]
        )
        candidate_positions = np.empty(len(feature_store), dtype=np.int64)
        for gene_index, values in enumerate(members):
            candidate_positions[np.asarray(values, dtype=np.int64)] = np.arange(
                len(values), dtype=np.int64
            )

        self.gene_ids = np.asarray(gene_ids, dtype=object)
        self.gene_to_index = gene_to_index
        self.offsets = offsets
        self.candidate_indices = candidate_indices
        self.candidate_gene_indices = candidate_gene_indices
        self.candidate_positions = candidate_positions

    def __len__(self) -> int:
        return len(self.gene_ids)

    def candidates(self, gene_index: int) -> np.ndarray:
        start = int(self.offsets[gene_index])
        end = int(self.offsets[gene_index + 1])
        return self.candidate_indices[start:end]


class AlignedGeneEmbeddingStore:
    """Memory-mapped gene embeddings aligned to a GeneCandidateIndex."""

    def __init__(
        self,
        embedding_file: str,
        gene_ids_file: str,
        availability_file: str,
        gene_index: GeneCandidateIndex,
    ) -> None:
        with open(gene_ids_file, "rt", encoding="utf-8") as handle:
            gene_ids = [line.rstrip("\n") for line in handle]
        if gene_ids != [str(value) for value in gene_index.gene_ids]:
            raise ValueError(
                "AgroNT gene IDs are not row-aligned to the candidate gene index"
            )
        self.embeddings = np.load(embedding_file, mmap_mode="r")
        self.available = np.load(availability_file, mmap_mode="r")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(gene_ids):
            raise ValueError("Invalid aligned AgroNT embedding matrix")
        if self.available.shape != (len(gene_ids),):
            raise ValueError("Invalid aligned AgroNT availability vector")
        self.embedding_dim = int(self.embeddings.shape[1])

    def model_inputs(self, gene_indices: np.ndarray) -> dict[str, torch.Tensor]:
        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        return {
            "gene_embeddings": torch.from_numpy(
                np.asarray(self.embeddings[gene_indices], dtype=np.float32)
            ),
            "gene_embedding_available": torch.from_numpy(
                np.asarray(self.available[gene_indices], dtype=np.bool_)
            ),
        }


@dataclass(frozen=True)
class ContextBatchLayout:
    instance_candidate_indices: np.ndarray
    instance_study_indices: np.ndarray
    instance_genotype_ids: np.ndarray
    instance_condition_ids: np.ndarray


class CandidateGeneContextMILCollator:
    """Expand each target candidate to its complete (gene, genotype) set."""

    def __init__(
        self,
        feature_store: CandidateFeatureStore,
        gene_index: GeneCandidateIndex,
        study_to_genotype: np.ndarray,
        num_genotypes: int,
        gene_embedding_store: AlignedGeneEmbeddingStore | None = None,
        num_conditions: int = 1,
    ) -> None:
        self.feature_store = feature_store
        self.gene_index = gene_index
        self.study_to_genotype = np.asarray(study_to_genotype, dtype=np.int64)
        self.num_genotypes = int(num_genotypes)
        self.num_conditions = int(num_conditions)
        self.gene_embedding_store = gene_embedding_store
        if self.study_to_genotype.ndim != 1:
            raise ValueError("study_to_genotype must be one-dimensional")
        if self.num_genotypes < 1:
            raise ValueError("num_genotypes must be positive")
        if self.num_conditions < 1:
            raise ValueError("num_conditions must be positive")

    def _context_batch(
        self,
        layout: ContextBatchLayout,
    ) -> dict[str, torch.Tensor]:
        candidate_indices = np.asarray(
            layout.instance_candidate_indices, dtype=np.int64
        )
        study_indices = np.asarray(layout.instance_study_indices, dtype=np.int64)
        genotype_ids = np.asarray(layout.instance_genotype_ids, dtype=np.int64)
        condition_ids = np.asarray(layout.instance_condition_ids, dtype=np.int64)
        if not (
            candidate_indices.ndim
            == study_indices.ndim
            == genotype_ids.ndim
            == condition_ids.ndim
            == 1
        ):
            raise ValueError("Instance arrays must be one-dimensional")
        if not (
            len(candidate_indices)
            == len(study_indices)
            == len(genotype_ids)
            == len(condition_ids)
        ):
            raise ValueError("Instance arrays must have equal length")
        if len(candidate_indices) == 0:
            raise ValueError("Cannot build an empty context batch")
        if int(genotype_ids.min()) < 0 or int(genotype_ids.max()) >= self.num_genotypes:
            raise ValueError("Instance genotype ID is outside the vocabulary")
        if (
            int(condition_ids.min()) < 0
            or int(condition_ids.max()) >= self.num_conditions
        ):
            raise ValueError("Instance condition ID is outside the vocabulary")

        instance_gene_indices = self.gene_index.candidate_gene_indices[
            candidate_indices
        ]
        group_keys = (
            (
                instance_gene_indices * self.num_genotypes
                + genotype_ids
            )
            * self.num_conditions
            + condition_ids
        )
        unique_group_keys, instance_group_indices = np.unique(
            group_keys, return_inverse=True
        )
        context_condition_ids = unique_group_keys % self.num_conditions
        genotype_gene_keys = unique_group_keys // self.num_conditions
        context_genotype_ids = genotype_gene_keys % self.num_genotypes
        context_gene_indices = genotype_gene_keys // self.num_genotypes

        group_members = [
            self.gene_index.candidates(int(gene_index))
            for gene_index in context_gene_indices
        ]
        group_lengths = np.asarray(
            [len(values) for values in group_members], dtype=np.int64
        )
        group_offsets = np.zeros(len(group_members) + 1, dtype=np.int64)
        group_offsets[1:] = np.cumsum(group_lengths)
        context_candidate_indices = np.concatenate(group_members)
        context_group_indices = np.repeat(
            np.arange(len(group_members), dtype=np.int64), group_lengths
        )
        context_position_indices = np.concatenate(
            [np.arange(length, dtype=np.int64) for length in group_lengths]
        )
        instance_context_indices = (
            group_offsets[instance_group_indices]
            + self.gene_index.candidate_positions[candidate_indices]
        )
        if not np.array_equal(
            context_candidate_indices[instance_context_indices],
            candidate_indices,
        ):
            raise RuntimeError("Candidate-to-gene context mapping is inconsistent")

        unique_candidate_indices, context_to_unique = np.unique(
            context_candidate_indices, return_inverse=True
        )
        batch = self.feature_store.model_inputs(unique_candidate_indices)
        batch.update(
            {
                "unique_candidate_indices": torch.from_numpy(
                    unique_candidate_indices
                ),
                "context_candidate_indices": torch.from_numpy(
                    context_candidate_indices
                ),
                "context_to_unique": torch.from_numpy(
                    context_to_unique.astype(np.int64, copy=False)
                ),
                "context_group_indices": torch.from_numpy(context_group_indices),
                "context_position_indices": torch.from_numpy(
                    context_position_indices
                ),
                "context_group_lengths": torch.from_numpy(group_lengths),
                "context_gene_indices": torch.from_numpy(
                    context_gene_indices.astype(np.int64, copy=False)
                ),
                "context_genotype_ids": torch.from_numpy(
                    context_genotype_ids.astype(np.int64, copy=False)
                ),
                "context_condition_ids": torch.from_numpy(
                    context_condition_ids.astype(np.int64, copy=False)
                ),
                "instance_candidate_indices": torch.from_numpy(candidate_indices),
                "instance_study_indices": torch.from_numpy(study_indices),
                "instance_genotype_ids": torch.from_numpy(genotype_ids),
                "instance_condition_ids": torch.from_numpy(condition_ids),
                "instance_context_indices": torch.from_numpy(
                    instance_context_indices.astype(np.int64, copy=False)
                ),
            }
        )
        if self.gene_embedding_store is not None:
            batch.update(
                self.gene_embedding_store.model_inputs(context_gene_indices)
            )
        return batch

    def build_context_batch(
        self,
        candidate_indices: np.ndarray,
        study_indices: np.ndarray | None = None,
        genotype_ids: np.ndarray | None = None,
        condition_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if (study_indices is None) == (genotype_ids is None):
            raise ValueError(
                "Provide exactly one of study_indices or genotype_ids"
            )
        if study_indices is not None:
            study_indices = np.asarray(study_indices, dtype=np.int64)
            genotype_ids = self.study_to_genotype[study_indices]
        else:
            genotype_ids = np.asarray(genotype_ids, dtype=np.int64)
            study_indices = np.full(len(candidate_indices), -1, dtype=np.int64)
        if condition_ids is None:
            condition_ids = np.zeros(len(candidate_indices), dtype=np.int64)
        else:
            condition_ids = np.asarray(condition_ids, dtype=np.int64)
        return self._context_batch(
            ContextBatchLayout(
                instance_candidate_indices=candidate_indices,
                instance_study_indices=study_indices,
                instance_genotype_ids=genotype_ids,
                instance_condition_ids=condition_ids,
            )
        )

    def __call__(self, examples: list[BagExample]) -> dict:
        if not examples:
            raise ValueError("Cannot collate an empty example list")
        instance_candidate_indices = np.concatenate(
            [example.candidate_indices for example in examples]
        ).astype(np.int64, copy=False)
        instance_bag_indices = np.concatenate(
            [
                np.full(len(example.candidate_indices), index, dtype=np.int64)
                for index, example in enumerate(examples)
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
        batch = self.build_context_batch(
            instance_candidate_indices,
            study_indices=instance_study_indices,
        )
        batch.update(
            {
                "instance_bag_indices": torch.from_numpy(instance_bag_indices),
                "labels": torch.tensor(
                    [example.label for example in examples],
                    dtype=torch.float32,
                ),
                "bag_sizes": torch.tensor(
                    [len(example.candidate_indices) for example in examples],
                    dtype=torch.float32,
                ),
                "bag_study_indices": torch.tensor(
                    [example.study_index for example in examples],
                    dtype=torch.long,
                ),
                "bag_region_ids": torch.tensor(
                    [example.region_id for example in examples],
                    dtype=torch.long,
                ),
                "sources": [example.source for example in examples],
                "source_indices": torch.tensor(
                    [example.source_index for example in examples],
                    dtype=torch.long,
                ),
            }
        )
        return batch
