# Expected demo outputs

These CSV files were generated from the real example inputs using the released inference checkpoints on 2026-10-17, with PyTorch 2.6.0+cu124 / Python 3.10.20 on CPU (one PyTorch thread).

- `candidate_predictions.csv`: 285 target-region adenosine probabilities.
- `quantitative_predictions.csv`: ten candidate predictions for AT5G06190 in the Col-0 / unstressed context.

`run_demo.py --verify` checks identifiers, row counts, finite values and numerical agreement (`rtol=1e-4`, `atol=1e-5`). The CSV files contain reference model predictions for the supplied example inputs.

Observed model-loading and inference time, excluding Python/PyTorch startup, was approximately 1.4 seconds for the candidate example and 4.4 seconds for the quantitative example on the reference host.
