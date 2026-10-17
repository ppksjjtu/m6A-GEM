# m6A-GEM

**Genetic–Environment Model of quantitative m6A landscapes.**

m6A-GEM contains two models for *Arabidopsis thaliana*: a sequence-based model that scores candidate adenosines, and a context-aware model that predicts quantitative m6A-associated signal using local sequence, AgroNT gene embeddings, genotype and stress information.

This repository provides the core architectures, their original training definitions, pretrained inference weights and small real-data inference examples. The repository is being prepared for manuscript submission; the examples demonstrate inference, not full manuscript reproduction.

## Quick start

Python 3.10 and a CPU are sufficient for the examples. From an environment with GitHub CLI access to this private repository:

```bash
gh repo clone ppksjjtu/m6A-GEM
cd m6A-GEM
python -m pip install -r requirements.txt
gh release download v0.1.0-models --repo ppksjjtu/m6A-GEM --dir weights --pattern '*.pt'
python examples/run_demo.py --model all --verify
```

If needed, run `gh auth login` first. You can also download the two `.pt` assets from the repository's **Releases** page and put them in `weights/`. Download access follows the repository's visibility.

The command writes `outputs/demo/candidate_predictions.csv` and `outputs/demo/quantitative_predictions.csv`, and checks them against the expected outputs included in the repository. CPU time and verification status are printed for each model. Individual examples can be run with `--model candidate` or `--model quantitative`.

## What is included

| Directory | Contents |
|---|---|
| `candidate_site/` | 13-block ConvMetaFormer candidate-site architecture, trainer and ROC/PR evaluation script |
| `quantitative/` | Local sequence Transformer, genotype–stress FiLM, gene-context Transformer, hurdle decoder, dataset definitions and trainer |
| `configs/` | Architecture, condition vocabulary and recorded training settings |
| `examples/` | Small real held-out inputs, inference code and expected predictions |
| `weights/` | Checkpoint manifest and download instructions; binary weights are Release assets |

The quantitative model is called `Stage1` in historical source filenames. In the manuscript's two-stage framework it is the second component, after candidate-site selection. The original model definitions and training objectives are preserved.

## Example inputs and outputs

The candidate-site example uses a strand-oriented chromosome 5 sequence window from `AT5G01010`. It outputs probabilities for adenosines inside the specified target region. `sequence_offset0` is a zero-based index in the supplied sequence, not a genomic coordinate.

The quantitative example retains all ten candidates from the held-out chromosome 5 gene `AT5G06190`, together with their 401-nt mature-transcript sequence windows, region and numerical features, and a 1,500-dimensional precomputed AgroNT gene embedding. It predicts the Col-0 / unstressed context. Details are in [`examples/data/README.json`](examples/data/README.json).

The quantitative decoder returns signal-presence probability and positive intensity. With the recorded `log1p` target transform:

```text
positive_intensity = expm1(softplus(intensity_logit))
expected_intensity = presence_probability * positive_intensity
```

Intensity is the model's prediction of non-negative IP-minus-background excess CPM. It is a signal proxy, not a methylation fraction. The two examples are independent model inputs; they do not implement a complete raw-sequencing-to-prediction pipeline.

## Training

For the original training scripts and their arguments:

```bash
python -m pip install -r requirements-training.txt
python candidate_site/29_train_atlas_rmbase_sac_support1_metaformer.py --help
python quantitative/stage1_copper_normal_singlehead_tailbalanced_v1.py --help
```

Training requires the complete processed data and appropriate hardware; those inputs are not bundled with the small examples. `configs/quantitative_checkpoint_settings.json` records the settings stored in the final Epoch 16 checkpoint after resumed training, rather than a complete first-launch history.

## Model weights and validation

The candidate-site checkpoint is from Epoch 8 and the quantitative checkpoint from Epoch 16. Release files contain the inference state and necessary configuration, with optimizer and gradient-scaler state removed. Network tensors are preserved. SHA-256 hashes are recorded in [`weights/pretrained_models.json`](weights/pretrained_models.json) and checked before inference.

The demo is intended to run with the reference PyTorch 2.6 / Python 3.10 environment. `--verify` compares all demo predictions with a saved CPU reference using `rtol=1e-4` and `atol=1e-5`. Full-data evaluation, manuscript analyses and additional preprocessing documentation will be added separately.

## License and citation

A software license and the manuscript citation will be added before public release. No publication DOI or archival DOI has been assigned in this repository.
