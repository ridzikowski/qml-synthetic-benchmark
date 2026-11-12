#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical MLP baseline for AE features, with any number of classes (>= 2).

Input CSV (same as the existing speech_ae_3f/4f/5f files):
    Feature_1,...,Feature_N,target,partition,speaker_id,sample_id

Partitions are speaker-disjoint:
    fit     -> train the MLP
    ae_val  -> choose the best checkpoint (validation cross-entropy)
    qnn_val -> exploratory evaluation ONLY; NOT an independent final test

Number of inputs = number of Feature_* columns.
Number of outputs = number of distinct target IDs in the CSV.
Any integer class IDs are accepted and internally remapped to 0..K-1.

Examples:
    python mlp_ae_baseline_multiclass.py --csv datasets/speech_ae_3f.csv
    python mlp_ae_baseline_multiclass.py --csv datasets/speech_ae_5f.csv --hidden 32
    python mlp_ae_baseline_multiclass.py --csv datasets/speech_ae_3f.csv --class-names zero one
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PARTITIONS = ("fit", "ae_val", "qnn_val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical multiclass MLP baseline for AE-exported speech features"
    )
    parser.add_argument("--csv", default="datasets/speech_ae_3f.csv", help="AE-export CSV path")
    parser.add_argument("--output-dir", default="out_mlp_ae", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=16, help="Neurons in a single hidden layer")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-names", nargs="+", default=None,
        help="Optional class names in ascending order of original target IDs",
    )
    parser.add_argument("--show", action="store_true", help="Also display matplotlib window")
    return parser.parse_args()


def _infer_class_names(df: pd.DataFrame, original_ids: list[int]) -> list[str]:
    """Infer human-readable words when sample_id is 'word/filename'."""
    fallback = [f"class_{target}" for target in original_ids]
    if "sample_id" not in df.columns:
        return fallback
    names: list[str] = []
    for target in original_ids:
        rows = df.loc[df["original_target"] == target, "sample_id"].dropna().astype(str)
        prefixes = set()
        for sample_id in rows:
            normalized = sample_id.replace("\\", "/")
            if "/" not in normalized:
                continue
            word = normalized.split("/", 1)[0].strip()
            word = re.sub(r"^\s*\d+\s*[._-]\s*", "", word).strip()
            if word:
                prefixes.add(word)
        names.append(next(iter(prefixes)) if len(prefixes) == 1 else f"class_{target}")
    # Do not silently assign ambiguous/duplicate names to distinct labels.
    if len(set(names)) != len(names):
        return fallback
    return names


def read_dataset(csv_path: Path, specified_names: list[str] | None = None):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cannot find dataset CSV: {csv_path}")

    # String dtype preserves leading zeros in speaker IDs.
    df = pd.read_csv(
        csv_path, dtype={"partition": "string", "speaker_id": "string", "sample_id": "string"}
    )
    feature_columns = sorted(
        (c for c in df.columns if re.fullmatch(r"Feature_\d+", c)),
        key=lambda c: int(c.split("_")[1]),
    )
    if not feature_columns:
        raise ValueError("CSV has no Feature_1, Feature_2, ... columns")
    expected = [f"Feature_{i + 1}" for i in range(len(feature_columns))]
    if feature_columns != expected:
        raise ValueError(
            f"Feature columns must be consecutive: expected {expected}, got {feature_columns}"
        )

    required = {"target", "partition", "speaker_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if "sample_id" in df.columns:
        if df["sample_id"].isna().any() or (df["sample_id"].str.strip() == "").any():
            raise ValueError("Empty or missing sample_id in dataset")
        if df["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample_id values detected")

    for column in ("partition", "speaker_id"):
        if df[column].isna().any() or (df[column].str.strip() == "").any():
            raise ValueError(f"Empty or missing {column} in dataset")
    if set(df["partition"].unique()) != set(PARTITIONS):
        raise ValueError(f"CSV must contain exactly the partitions: {PARTITIONS}")

    df = df.copy()
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[feature_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("AE features contain NaN or infinite values")

    raw = pd.to_numeric(df["target"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("target must contain finite integer class IDs")
    if (raw < np.iinfo(np.int64).min).any() or (raw >= 2**63).any():
        raise ValueError("target IDs are outside int64 range")
    df["original_target"] = raw.astype(np.int64)

    original_ids = sorted(int(label) for label in df["original_target"].unique())
    if len(original_ids) < 2:
        raise ValueError("At least two classes are required for classification")
    to_internal = {label: index for index, label in enumerate(original_ids)}
    df["target_internal"] = df["original_target"].map(to_internal).astype(np.int64)

    if specified_names is not None:
        if len(specified_names) != len(original_ids):
            raise ValueError(
                f"--class-names requires exactly {len(original_ids)} names "
                f"(original targets: {original_ids})"
            )
        if len(set(specified_names)) != len(specified_names):
            raise ValueError("--class-names must contain unique names")
        class_names = specified_names
    else:
        class_names = _infer_class_names(df, original_ids)

    result = {}
    speaker_groups = {}
    all_classes = set(range(len(original_ids)))
    for name in PARTITIONS:
        block = df.loc[df["partition"] == name]
        x = block[feature_columns].to_numpy(dtype=np.float32)
        y = block["target_internal"].to_numpy(dtype=np.int64)
        if len(x) == 0:
            raise ValueError(f"Partition {name} has no samples")
        present = set(int(v) for v in np.unique(y))
        missing_classes = all_classes - present
        if missing_classes:
            missing_text = [class_names[i] for i in sorted(missing_classes)]
            if name == "fit":
                raise ValueError(f"Training partition fit is missing classes: {missing_text}")
            warnings.warn(
                f"Partition {name} has no samples of classes {missing_text}; "
                "report will still include all classes.",
                stacklevel=2,
            )
        result[name] = (x, y)
        speaker_groups[name] = set(block["speaker_id"].astype(str))

    for a, b in (("fit", "ae_val"), ("fit", "qnn_val"), ("ae_val", "qnn_val")):
        shared = speaker_groups[a] & speaker_groups[b]
        if shared:
            raise ValueError(f"Speaker leakage between {a} and {b}: {sorted(shared)[:3]}")

    metadata = {
        "original_target_ids": original_ids,
        "class_names": class_names,
        "num_classes": len(original_ids),
        "internal_to_original": {i: label for i, label in enumerate(original_ids)},
        "df": df,
    }
    return result, feature_columns, metadata


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, loss_fn: nn.Module):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        loss = float(loss_fn(logits, torch.from_numpy(y)).item())
        predictions = logits.argmax(dim=1).cpu().numpy()
    return loss, float(accuracy_score(y, predictions)), predictions


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.hidden, args.batch_size) <= 0 or args.lr <= 0:
        raise ValueError("epochs, patience, hidden, batch-size and lr must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions, features, meta = read_dataset(csv_path, args.class_names)
    class_names = meta["class_names"]
    original_ids = meta["original_target_ids"]
    num_classes = meta["num_classes"]
    labels = list(range(num_classes))
    x_train, y_train = partitions["fit"]
    x_val, y_val = partitions["ae_val"]
    x_eval, y_eval = partitions["qnn_val"]

    print(f"Features: {features}")
    print(f"Classes ({num_classes}): " + ", ".join(
        f"{id_}={name}" for id_, name in zip(original_ids, class_names)
    ))
    print(f"Train (fit): {len(y_train)}; validation (ae_val): {len(y_val)}; "
          f"exploratory qnn_val: {len(y_eval)}")

    model = nn.Sequential(
        nn.Linear(len(features), args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, num_classes),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _ = evaluate(model, x_train, y_train, loss_fn)
        val_loss, val_acc, _ = evaluate(model, x_val, y_val, loss_fn)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epochs_without_improvement >= args.patience:
            print(f"epoch={epoch:03d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
                  f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)

    train_loss, train_acc, _ = evaluate(model, x_train, y_train, loss_fn)
    val_loss, val_acc, _ = evaluate(model, x_val, y_val, loss_fn)
    eval_loss, eval_acc, eval_predictions = evaluate(model, x_eval, y_eval, loss_fn)
    eval_macro_f1 = float(f1_score(
        y_eval, eval_predictions, labels=labels, average="macro", zero_division=0
    ))
    text_report = classification_report(
        y_eval, eval_predictions, labels=labels, target_names=class_names,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_eval, eval_predictions, labels=labels).tolist()
    class_legend = ", ".join(f"{id_}={name}" for id_, name in zip(original_ids, class_names))

    print(f"\nBest epoch: {best_epoch}")
    print(f"Exploratory qnn_val accuracy: {eval_acc:.4f}; macro-F1: {eval_macro_f1:.4f}; "
          f"loss: {eval_loss:.4f}")
    print("Classification report (qnn_val, exploratory):\n")
    print(text_report)
    print(f"Confusion matrix (rows=true, columns=predicted; {class_legend}):")
    for row in cm:
        print(row)

    (output_dir / "mlp_classification_report.txt").write_text(
        f"Best checkpoint epoch: {best_epoch}\n"
        f"Data split: train=fit, val=ae_val, evaluation=qnn_val\n"
        f"Classes: {class_legend}\n"
        f"Evaluation accuracy: {eval_acc:.6f}\n"
        f"Evaluation macro-F1: {eval_macro_f1:.6f}\n"
        f"Evaluation loss: {eval_loss:.6f}\n\n"
        f"{text_report}\nConfusion matrix (rows=true):\n"
        + "\n".join(str(row) for row in cm) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(output_dir / "mlp_history.csv", index=False)

    # Per-sample exploratory predictions preserve IDs for later QNN comparisons.
    eval_block = meta["df"].loc[meta["df"]["partition"] == "qnn_val"].copy()
    prediction_table = pd.DataFrame({
        "sample_id": eval_block["sample_id"].to_numpy() if "sample_id" in eval_block else [""] * len(eval_block),
        "speaker_id": eval_block["speaker_id"].astype(str).to_numpy(),
        "target": eval_block["original_target"].to_numpy(dtype=np.int64),
        "prediction": np.asarray([original_ids[i] for i in eval_predictions], dtype=np.int64),
    })
    prediction_table["correct"] = prediction_table["target"] == prediction_table["prediction"]
    prediction_table.to_csv(output_dir / "mlp_qnn_val_predictions.csv", index=False)

    json_report = {
        "status": "completed",
        "dataset_csv": str(csv_path),
        "features": features,
        "num_classes": num_classes,
        "class_ids": original_ids,
        "class_names": class_names,
        "class_map": {str(raw): {"internal_id": i, "name": class_names[i]}
                      for i, raw in enumerate(original_ids)},
        "seed": args.seed,
        "hidden": args.hidden,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "sample_counts": {key: len(value[1]) for key, value in partitions.items()},
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "qnn_val_accuracy_exploratory": eval_acc,
        "qnn_val_macro_f1_exploratory": eval_macro_f1,
        "qnn_val_loss_exploratory": eval_loss,
        "classification_report_qnn_val_exploratory": classification_report(
            y_eval, eval_predictions, labels=labels,
            target_names=class_names, output_dict=True, zero_division=0,
        ),
        "confusion_matrix_rows_true": cm,
        "scope_note": "qnn_val is exploratory and not an independent final test",
    }
    # Backward-compatible JSON field for original binary experiments.
    if original_ids == [0, 1] and class_names == ["zero", "one"]:
        json_report["confusion_matrix_rows_true_zero_one"] = cm
    (output_dir / "mlp_report.json").write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": len(features),
        "hidden_dim": args.hidden,
        "num_classes": num_classes,
        "class_ids": original_ids,
        "class_names": class_names,
        "features": features,
        "seed": args.seed,
    }, output_dir / "mlp_best.pt")

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(epochs, [h["train_acc"] for h in history], label="Train (fit)")
    axes[0].plot(epochs, [h["val_acc"] for h in history], label="Validation (ae_val)")
    axes[0].axvline(best_epoch, linestyle="--", alpha=0.65, label=f"Best epoch {best_epoch}")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="MLP accuracy vs epoch")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, [h["train_loss"] for h in history], label="Train (fit)")
    axes[1].plot(epochs, [h["val_loss"] for h in history], label="Validation (ae_val)")
    axes[1].axvline(best_epoch, linestyle="--", alpha=0.65, label=f"Best epoch {best_epoch}")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="MLP loss vs epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"Classical MLP on {csv_path.name} | {num_classes} classes | "
                 f"qnn_val acc={eval_acc:.3f} (exploratory)")
    fig.tight_layout()
    plot_path = output_dir / "mlp_accuracy_loss.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved files in: {output_dir}")
    print(f"Plot: {plot_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
