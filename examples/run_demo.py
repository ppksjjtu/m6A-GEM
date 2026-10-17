#!/usr/bin/env python3
"""Run the two pretrained m6A-GEM models on small real held-out inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from _model_utils import ROOT, load_model


def candidate_demo(model):
    sample = json.loads((ROOT/'examples/data/candidate_window.json').read_text())
    sequence = sample['seq'].upper()
    vocab = {'A':0,'C':1,'G':2,'T':3,'U':3,'N':4}
    tokens = torch.tensor([[vocab.get(base,4) for base in sequence]],dtype=torch.long)
    valid = torch.ones_like(tokens,dtype=torch.bool)
    peak = torch.zeros_like(valid)
    start, end = sample['peak_start_in_seq'], sample['peak_end_in_seq']
    peak[:,start:end] = True
    with torch.inference_mode():
        logits, _, _, _ = model(tokens,valid,peak)
        scores = torch.sigmoid(logits[0]).numpy()
    return [{
        'sample_id':sample['sample_id'], 'sequence_offset0':str(i),
        'candidate_probability':format(float(scores[i]),'.9g'),
    } for i in range(start,end) if sequence[i]=='A']


def quantitative_demo(model,checkpoint):
    data = np.load(ROOT/'examples/data/quantitative_gene.npz',allow_pickle=False)
    with (ROOT/'examples/data/quantitative_candidates.tsv').open() as handle:
        metadata = list(csv.DictReader(handle,delimiter='\t'))
    n = len(metadata)
    assert data['sequence_tokens'].shape == (n,401)
    batch = {
        'sequence_tokens':torch.from_numpy(data['sequence_tokens'].copy()),
        'region_ids':torch.from_numpy(data['region_ids'].copy()),
        'numeric_features':torch.from_numpy(data['numeric_features'].copy()),
        'context_to_unique':torch.arange(n),
        'context_genotype_ids':torch.tensor([int(data['genotype_id'])]),
        'context_condition_ids':torch.tensor([int(data['condition_id'])]),
        'context_group_indices':torch.zeros(n,dtype=torch.long),
        'context_group_lengths':torch.tensor([n]),
        'context_position_indices':torch.arange(n),
        'gene_embeddings':torch.from_numpy(data['gene_embedding'].copy()).unsqueeze(0),
        'gene_embedding_available':torch.tensor([bool(data['gene_embedding_available'])]),
        'instance_context_indices':torch.arange(n),
    }
    with torch.inference_mode():
        output = model.forward_intensity(batch)
        probability = output['presence_probability'].float()
        positive = output['positive_intensity_target_space'].float()
        if checkpoint['args']['target_transform']=='log1p':
            positive = torch.expm1(positive).clamp_min(0)
        expected = probability*positive
    assert torch.isfinite(expected).all()
    return [{
        'candidate_id':row['candidate_id'],
        'presence_probability':format(float(probability[i]),'.9g'),
        'positive_intensity':format(float(positive[i]),'.9g'),
        'expected_intensity':format(float(expected[i]),'.9g'),
    } for i,row in enumerate(metadata)]


def verify(actual,reference:Path,kind:str):
    with reference.open() as handle:
        expected = list(csv.DictReader(handle))
    if len(actual)!=len(expected):
        raise AssertionError('Output row count differs from expected results')
    numeric = ['candidate_probability'] if kind=='candidate' else ['presence_probability','positive_intensity','expected_intensity']
    identifier = ['sample_id','sequence_offset0'] if kind=='candidate' else ['candidate_id']
    for a,e in zip(actual,expected):
        if any(a[key]!=e[key] for key in identifier):
            raise AssertionError('Output identifiers differ from expected results')
    for key in numeric:
        values = np.asarray([float(row[key]) for row in actual])
        targets = np.asarray([float(row[key]) for row in expected])
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            raise AssertionError(f'Non-finite output in {key}')
        np.testing.assert_allclose(values,targets,rtol=1e-4,atol=1e-5,err_msg=key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['candidate','quantitative','all'],default='all')
    parser.add_argument('--weights-dir',type=Path,default=ROOT/'weights')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/demo')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.model=='all':
        for kind in ['candidate','quantitative']:
            command=[sys.executable,str(Path(__file__).resolve()),'--model',kind,'--weights-dir',str(args.weights_dir.resolve()),'--output-dir',str(args.output_dir.resolve())]
            if args.verify:command.append('--verify')
            subprocess.run(command,check=True)
        return
    torch.set_num_threads(1)
    start=time.perf_counter()
    model,checkpoint=load_model(args.model,args.weights_dir)
    rows=candidate_demo(model) if args.model=='candidate' else quantitative_demo(model,checkpoint)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    filename=args.model+'_predictions.csv'
    target=args.output_dir/filename
    with target.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if args.verify:verify(rows,ROOT/'examples/expected'/filename,args.model)
    print(json.dumps({'model':args.model,'n_predictions':len(rows),'best_epoch':checkpoint['best_epoch'],'output':str(target),'verification':'PASS' if args.verify else 'not requested','cpu_seconds':round(time.perf_counter()-start,3)}))


if __name__=='__main__':
    main()
