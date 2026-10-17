#!/usr/bin/env python3
"""Evaluate the best Stage 1 checkpoint and plot true Chr5 ROC/PR curves."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_training_module():
    path = ROOT / "29_train_atlas_rmbase_sac_support1_metaformer.py"
    spec = importlib.util.spec_from_file_location("stage1_training", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def collect_scores(model, loader, device, amp):
    labels, scores = [], []
    model.eval()
    for batch in loader:
        sequence = batch["seq"].to(device, non_blocking=True)
        input_mask = batch["input_mask"].to(device, non_blocking=True)
        peak_mask = batch["peak_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        a_mask = batch["a_mask"].to(device, non_blocking=True).bool()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            site_logits, _, _, _ = model(sequence, input_mask, peak_mask)
        labels.append(targets[a_mask].cpu().numpy().astype(np.uint8))
        scores.append(torch.sigmoid(site_logits[a_mask]).cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_curve(x, y, xlabel, ylabel, legend, output_stem, baseline=None):
    fig, ax = plt.subplots(figsize=(3.25, 3.0))
    ax.plot(x, y, color="#315E8A", linewidth=1.8, label=legend)
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#9AA3AC",
            linewidth=0.9,
            linestyle="--",
            label=f"Prevalence = {baseline:.3f}",
        )
    elif xlabel.startswith("False"):
        ax.plot([0, 1], [0, 1], color="#9AA3AC", linewidth=0.9, linestyle="--")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#DDE2E7", linewidth=0.5, alpha=0.75)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.8)
    for extension, kwargs in {
        ".png": {"dpi": 600},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(output_stem.with_suffix(extension), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/site_peak1000_atlas_rmbase_sac_support1/"
        "m6a_site_peakmax1000_atlas_rmbase_sac_support1.tsv.gz",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/site_peak1000_atlas_rmbase_sac_support1_d192_l13_run1/"
        "best_model.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="results/site_peak1000_atlas_rmbase_sac_support1_d192_l13_run1/curves",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    training = load_training_module()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["args"]
    requested_cuda = args.device.startswith("cuda")
    device = torch.device(
        args.device if requested_cuda and torch.cuda.is_available() else "cpu"
    )
    frame = pd.read_csv(args.dataset, sep="\t", low_memory=False)
    test = frame.loc[frame["split"].eq("test")].copy()
    loader = DataLoader(
        training.WindowDataset(test, int(config.get("seq_len", 1400))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=training.collate,
    )
    model = training.M6ASiteSACMetaFormer(
        seq_len=int(config.get("seq_len", 1400)),
        d_model=int(config.get("d_model", 192)),
        n_layers=int(config.get("n_layers", 13)),
        kernel_size=int(config.get("kernel_size", 9)),
        dropout=float(config.get("dropout", 0.15)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    labels, scores = collect_scores(model, loader, device, args.amp)
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics = {
        "split": "Chr5 test",
        "n_A": int(labels.size),
        "n_positive_A": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc_average_precision": float(average_precision_score(labels, scores)),
        "auprc_trapezoidal": float(auc(recall[::-1], precision[::-1])),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "chr5_test_A_site_labels_scores.npz", labels=labels, scores=scores)
    (out_dir / "curve_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    style()
    save_curve(
        fpr,
        tpr,
        "False-positive rate",
        "True-positive rate",
        f"Chr5 test (AUROC = {metrics['auroc']:.3f})",
        out_dir / "chr5_test_A_site_ROC",
    )
    save_curve(
        recall,
        precision,
        "Recall",
        "Precision",
        f"Chr5 test (AUPRC = {metrics['auprc_average_precision']:.3f})",
        out_dir / "chr5_test_A_site_PR",
        baseline=metrics["positive_rate"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

