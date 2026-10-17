# Pretrained inference weights

Download the two assets from the `v0.1.0-models` release:

```bash
gh release download v0.1.0-models --repo ppksjjtu/m6A-GEM --dir weights --pattern '*.pt'
```

- `candidate_site_epoch8.pt`: candidate-site model.
- `quantitative_epoch16.pt`: context-aware quantitative model.

`pretrained_models.json` contains model identifiers and SHA-256 hashes. The demo verifies each file and loads the pretrained model tensors.
