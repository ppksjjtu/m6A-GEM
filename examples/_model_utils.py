"""Load m6A-GEM inference checkpoints without importing the training pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_model(kind: str, weights_dir: Path):
    filename = 'candidate_site_epoch8.pt' if kind == 'candidate' else 'quantitative_epoch16.pt'
    path = weights_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is missing. Download the v0.1.0-models release assets as described in README.md.'
        )
    manifest = json.loads((ROOT/'weights/pretrained_models.json').read_text())
    expected = manifest['models'][filename]['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Checkpoint SHA-256 mismatch: {filename}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['model_type'] != kind:
        raise ValueError('Checkpoint is for a different model')
    state, settings = checkpoint['model_state'], checkpoint['args']
    # Run the two stages in separate processes: each retains its original models package.
    sys.path.insert(0, str(ROOT/('candidate_site' if kind == 'candidate' else 'quantitative')))
    if kind == 'candidate':
        from models.m6a_site_sac_metaformer import build_model_from_config
        model = build_model_from_config(settings)
    else:
        from models.m6a_candidate_intensity_singlehead_hurdle_model import CandidateIntensitySingleHeadHurdleModel
        qc, protocol = state['qc_features'], state['protocol_ids']
        sequence_dim, region_dim = settings['sequence_encoder_dim'], settings['region_embedding_dim']
        model = CandidateIntensitySingleHeadHurdleModel(
            num_studies=int(qc.shape[0]),
            num_genotypes=int(state['genotype_embedding.weight'].shape[0]),
            num_conditions=int(state['condition_embedding.weight'].shape[0]),
            condition_reference_id=checkpoint['condition_vocab']['unstressed'],
            num_protocols=int(protocol.max())+1,
            qc_features=qc, protocol_ids=protocol,
            gene_embedding_dim=int(state['gene_projection.0.weight'].numel()),
            sequence_length=int(state['sequence_encoder.position_embedding'].shape[1])-1,
            sequence_encoder_dim=sequence_dim,
            sequence_encoder_layers=settings['sequence_encoder_layers'],
            sequence_encoder_kernel_size=7,
            sequence_encoder_dropout=settings['dropout'],
            sequence_chunk_size=settings['sequence_chunk_size'],
            sequence_transformer_heads=settings['sequence_transformer_heads'],
            sequence_transformer_ffn_multiplier=settings['sequence_transformer_ffn_multiplier'],
            sequence_gradient_checkpointing=False,
            num_regions=int(state['region_embedding.weight'].shape[0]),
            region_embedding_dim=region_dim,
            numeric_feature_dim=int(state['candidate_projection.0.weight'].numel())-sequence_dim-region_dim,
            hidden_dim=settings['hidden_dim'],
            transformer_layers=settings['transformer_layers'],
            transformer_heads=settings['transformer_heads'],
            transformer_ffn_multiplier=settings['transformer_ffn_multiplier'],
            gene_token_count=settings['gene_token_count'], dropout=settings['dropout'],
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint
