# Pretrained inference weights

Download the two assets from the `v0.1.0-models` release:

```bash
gh release download v0.1.0-models --repo ppksjjtu/m6A-GEM --dir weights --pattern '*.pt'
```

- `candidate_site_epoch8.pt`: candidate-site model.
- `quantitative_epoch16.pt`: context-aware quantitative model.

`pretrained_models.json` contains asset SHA-256 hashes, source-checkpoint hashes and model identifiers. The demo verifies hashes and strictly loads model tensors. These files are inference checkpoints; optimizer state was removed, so they are not full training-resume archives.
