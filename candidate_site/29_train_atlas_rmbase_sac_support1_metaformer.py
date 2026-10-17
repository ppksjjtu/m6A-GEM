#!/usr/bin/env python3
"""Train the modular d192 m6A A-site model on Atlas/RMBase + SAC-seq."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from models.m6a_site_sac_metaformer import (  # noqa: E402
    M6ASiteSACMetaFormer,
    PAD_TOKEN,
)


VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
IGNORE_LABEL = -100.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_json_array(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    return json.loads(str(value))


def encode_sequence(sequence, maximum_length):
    sequence = str(sequence).upper()[:maximum_length]
    return np.asarray([VOCAB.get(base, VOCAB["N"]) for base in sequence], dtype=np.uint8)


def source_code(value):
    sources = set(str(value).split("+"))
    has_old = bool({"atlas", "rmbase"}.intersection(sources))
    has_sac = "sac" in sources
    return int(has_old) + 2 * int(has_sac)


class WindowDataset(Dataset):
    def __init__(self, frame, seq_len):
        self.frame = frame.reset_index(drop=True)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        sequence = encode_sequence(row["seq"], self.seq_len)
        length = len(sequence)
        peak_mask = np.zeros(length, dtype=np.bool_)
        peak_start = int(row["peak_start_in_seq"])
        peak_end = min(int(row["peak_end_in_seq"]), length)
        peak_mask[peak_start:peak_end] = True

        labels = np.full(length, IGNORE_LABEL, dtype=np.float32)
        a_mask = np.zeros(length, dtype=np.bool_)
        label_weights = np.zeros(length, dtype=np.float32)
        source_codes = np.zeros(length, dtype=np.int8)
        sac_support = np.zeros(length, dtype=np.int8)
        positions = parse_json_array(row["a_positions_json"])
        targets = parse_json_array(row["m6a_labels_json"])
        weights = parse_json_array(row["label_weights_json"])
        sources = parse_json_array(row.get("positive_sources_json", ""))
        supports = parse_json_array(row.get("positive_sac_support_json", ""))
        if not (len(positions) == len(targets) == len(weights)):
            raise ValueError(f"Malformed label arrays at row {index}")
        if len(sources) != len(targets):
            sources = [""] * len(targets)
        if len(supports) != len(targets):
            supports = [0] * len(targets)
        for position, target, weight, source, support in zip(
            positions, targets, weights, sources, supports
        ):
            position = int(position)
            if 0 <= position < length:
                labels[position] = float(target)
                a_mask[position] = True
                label_weights[position] = float(weight)
                source_codes[position] = source_code(source) if target else 0
                sac_support[position] = int(support) if target else 0
        return {
            "seq": sequence,
            "peak_mask": peak_mask,
            "labels": labels,
            "a_mask": a_mask,
            "label_weights": label_weights,
            "source_codes": source_codes,
            "sac_support": sac_support,
            "has_m6a": float(row["has_m6A"]),
        }


def collate(items):
    batch_size = len(items)
    maximum_length = max(len(item["seq"]) for item in items)
    seq = np.full((batch_size, maximum_length), PAD_TOKEN, dtype=np.int64)
    input_mask = np.zeros((batch_size, maximum_length), dtype=np.bool_)
    peak_mask = np.zeros((batch_size, maximum_length), dtype=np.bool_)
    labels = np.full((batch_size, maximum_length), IGNORE_LABEL, dtype=np.float32)
    a_mask = np.zeros((batch_size, maximum_length), dtype=np.bool_)
    weights = np.zeros((batch_size, maximum_length), dtype=np.float32)
    source_codes = np.zeros((batch_size, maximum_length), dtype=np.int8)
    sac_support = np.zeros((batch_size, maximum_length), dtype=np.int8)
    for index, item in enumerate(items):
        length = len(item["seq"])
        seq[index, :length] = item["seq"]
        input_mask[index, :length] = True
        peak_mask[index, :length] = item["peak_mask"]
        labels[index, :length] = item["labels"]
        a_mask[index, :length] = item["a_mask"]
        weights[index, :length] = item["label_weights"]
        source_codes[index, :length] = item["source_codes"]
        sac_support[index, :length] = item["sac_support"]
    return {
        "seq": torch.from_numpy(seq),
        "input_mask": torch.from_numpy(input_mask),
        "peak_mask": torch.from_numpy(peak_mask),
        "labels": torch.from_numpy(labels),
        "a_mask": torch.from_numpy(a_mask),
        "label_weights": torch.from_numpy(weights),
        "source_codes": torch.from_numpy(source_codes),
        "sac_support": torch.from_numpy(sac_support),
        "has_m6a": torch.tensor(
            [item["has_m6a"] for item in items], dtype=torch.float32
        ),
    }


def weighted_label_counts(frame):
    positive_weight = 0.0
    negative_weight = 0.0
    raw_positive = 0
    raw_total = 0
    for labels_json, weights_json in zip(
        frame["m6a_labels_json"], frame["label_weights_json"]
    ):
        labels = parse_json_array(labels_json)
        weights = parse_json_array(weights_json)
        raw_positive += int(sum(labels))
        raw_total += len(labels)
        for label, weight in zip(labels, weights):
            if label:
                positive_weight += float(weight)
            else:
                negative_weight += float(weight)
    return raw_positive, raw_total, positive_weight, negative_weight


def masked_site_loss(logits, labels, a_mask, weights, pos_weight):
    valid = a_mask.bool()
    losses = F.binary_cross_entropy_with_logits(
        logits[valid], labels[valid], pos_weight=pos_weight, reduction="none"
    )
    valid_weights = weights[valid]
    return (losses * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)


def binary_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    result = {
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
    }
    if len(np.unique(labels)) == 2:
        result.update(
            {
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                "mcc": float(matthews_corrcoef(labels, predictions)),
            }
        )
    else:
        result.update({"auroc": float("nan"), "auprc": float("nan"), "mcc": float("nan")})
    return result


@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches=0):
    model.eval()
    site_labels, site_scores = [], []
    window_labels, window_scores = [], []
    source_codes, sac_supports = [], []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        sequence = batch["seq"].to(device, non_blocking=True)
        input_mask = batch["input_mask"].to(device, non_blocking=True)
        peak_mask = batch["peak_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        a_mask = batch["a_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            site_logits, window_logits, _, _ = model(sequence, input_mask, peak_mask)
        valid = a_mask.bool()
        site_labels.append(labels[valid].cpu().numpy())
        site_scores.append(torch.sigmoid(site_logits[valid]).cpu().numpy())
        source_codes.append(batch["source_codes"][valid.cpu()].numpy())
        sac_supports.append(batch["sac_support"][valid.cpu()].numpy())
        window_labels.append(batch["has_m6a"].numpy())
        window_scores.append(torch.sigmoid(window_logits).cpu().numpy())

    y_site = np.concatenate(site_labels)
    p_site = np.concatenate(site_scores)
    y_window = np.concatenate(window_labels)
    p_window = np.concatenate(window_scores)
    sources = np.concatenate(source_codes)
    supports = np.concatenate(sac_supports)
    result = {
        "A_site": binary_metrics(y_site, p_site),
        "window": binary_metrics(y_window, p_window),
        "positive_source_recall_at_0p5": {},
        "sac_support_recall_at_0p5": {},
    }
    positive = y_site == 1
    for code, label in ((1, "old_only"), (2, "sac_only"), (3, "old_and_sac")):
        mask = positive & (sources == code)
        result["positive_source_recall_at_0p5"][label] = {
            "n": int(mask.sum()),
            "recall": float((p_site[mask] >= 0.5).mean()) if mask.any() else None,
        }
    for minimum_support in (1, 2, 3, 5, 9):
        mask = positive & (supports >= minimum_support)
        result["sac_support_recall_at_0p5"][f"support_ge_{minimum_support}"] = {
            "n": int(mask.sum()),
            "recall": float((p_site[mask] >= 0.5).mean()) if mask.any() else None,
        }
    return result


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    pos_weight,
    auxiliary_weight,
    amp,
    max_batches=0,
):
    model.train()
    losses = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        sequence = batch["seq"].to(device, non_blocking=True)
        input_mask = batch["input_mask"].to(device, non_blocking=True)
        peak_mask = batch["peak_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        a_mask = batch["a_mask"].to(device, non_blocking=True)
        weights = batch["label_weights"].to(device, non_blocking=True)
        has_m6a = batch["has_m6a"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            site_logits, window_logits, _, _ = model(sequence, input_mask, peak_mask)
            site_loss = masked_site_loss(
                site_logits, labels, a_mask, weights, pos_weight
            )
            window_loss = F.binary_cross_entropy_with_logits(window_logits, has_m6a)
            loss = site_loss + auxiliary_weight * window_loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/site_peak1000_atlas_rmbase_sac_support1/m6a_site_peakmax1000_atlas_rmbase_sac_support1.tsv.gz",
    )
    parser.add_argument(
        "--out-dir", default="results/site_peak1000_atlas_rmbase_sac_support1_d192_l13_run1"
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--seq-len", type=int, default=1400)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=13)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--window-aux-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Debug only: stop each training epoch after this many batches; 0 uses all batches.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Debug only: stop each evaluation pass after this many batches; 0 uses all batches.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    frame = pd.read_csv(args.dataset, sep="\t", low_memory=False)
    train_frame = frame.loc[frame["split"].eq("train")].copy()
    val_frame = frame.loc[frame["split"].eq("val")].copy()
    test_frame = frame.loc[frame["split"].eq("test")].copy()
    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("Train/validation/test split contains an empty partition")

    raw_positive, raw_total, weighted_positive, weighted_negative = weighted_label_counts(
        train_frame
    )
    pos_weight_value = min(weighted_negative / max(weighted_positive, 1.0), 30.0)
    print(
        json.dumps(
            {
                "dataset_rows": len(frame),
                "split_rows": {
                    "train": len(train_frame),
                    "val": len(val_frame),
                    "test": len(test_frame),
                },
                "train_A_labels": raw_total,
                "train_positive_A": raw_positive,
                "pos_weight": pos_weight_value,
            },
            indent=2,
        )
    )

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        WindowDataset(train_frame, args.seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        WindowDataset(val_frame, args.seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        WindowDataset(test_frame, args.seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    requested_cuda = args.device.startswith("cuda")
    device = torch.device(args.device if requested_cuda and torch.cuda.is_available() else "cpu")
    if requested_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was requested but no GPU is available")
    model = M6ASiteSACMetaFormer(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        print(f"[initialized] {args.init_checkpoint}")
    print(f"[device] {device} [parameters] {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

    history = []
    best_auprc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            pos_weight,
            args.window_aux_weight,
            args.amp,
            args.max_train_batches,
        )
        validation = evaluate(
            model, val_loader, device, args.amp, args.max_eval_batches
        )
        val_auprc = validation["A_site"]["auprc"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_A_site_auprc": val_auprc,
            "val_A_site_auroc": validation["A_site"]["auroc"],
            "val_A_site_mcc": validation["A_site"]["mcc"],
            "val_window_auprc": validation["window"]["auprc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "epoch_metrics.tsv", sep="\t", index=False)
        print(json.dumps(row))
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_A_site_auprc": best_auprc,
                },
                out_dir / "best_model.pt",
            )
            (out_dir / "best_validation_metrics.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            print(f"[early stopping] epoch={epoch}")
            break

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    final = {
        "best_epoch": best_epoch,
        "best_val_A_site_auprc": best_auprc,
        "train": evaluate(
            model, train_loader, device, args.amp, args.max_eval_batches
        ),
        "validation": evaluate(
            model, val_loader, device, args.amp, args.max_eval_batches
        ),
        "test": evaluate(
            model, test_loader, device, args.amp, args.max_eval_batches
        ),
    }
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
