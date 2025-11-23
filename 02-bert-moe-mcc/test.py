#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py — finds a checkpoint (or uses provided one), reconstructs the model from checkpoint hparams/state,
evaluates on a specified test dataset, and writes per-class CSV + confusion matrix figure into an output folder.

Key Features:
- Automatically picks the best local checkpoint if none is provided.
- Computes detailed per-class metrics and confusion matrix.

Usage:
  python test.py --name myrun --data_dir ../data/imdb_arh_synthetic --checkpoint_path checkpoints/...
"""
import argparse
import re
from pathlib import Path
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

try:
    torch.set_float32_matmul_precision('medium')
except Exception:
    pass

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
)
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from utils.module import MoE_LightningModule, IMDBDataset
from layers.encoder import SBERT_MoE_Model

# ---------------------------------------------------------------------
def pick_best_local_checkpoint(checkpoints_root="checkpoints/multiclass"):
    root = Path(checkpoints_root)
    ckpt_paths = list(root.rglob("*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_root}")
    best_val = -1.0
    best_path = None
    for p in ckpt_paths:
        m = re.search(r"val_acc=([0-9]*\.?[0-9]+)", p.name)
        val = float(m.group(1)) if m else p.stat().st_mtime
        if val > best_val:
            best_val = val
            best_path = p
    return str(best_path)

# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=f"test_{int(time.time())}", help="Name for this test run (subfolder)")
    parser.add_argument("--data_dir", type=str, default="../data/imdb_arh_trimmed", help="Path to data directory")
    parser.add_argument("--train_file", type=str, default="imdb_arh_train.csv", help="Train CSV filename (used to discover classes/tokenizer)")
    parser.add_argument("--test_file", type=str, default="imdb_arh_test.csv", help="Test CSV filename")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint .ckpt file; if omitted the best local is used")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Directory to write outputs into")
    return parser.parse_args()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # resolve checkpoint
    if args.checkpoint_path:
        CHECKPOINT_PATH = args.checkpoint_path
    else:
        CHECKPOINT_PATH = pick_best_local_checkpoint("checkpoints/multiclass")
    print(f"✅ Using checkpoint: {CHECKPOINT_PATH}")

    # dataset & loader
    DATA_DIR = Path(args.data_dir)
    TRAIN_FILE = args.train_file
    TEST_FILE = args.test_file
    BATCH_SIZE = args.batch_size

    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, TRAIN_FILE)
    print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

    tr_ds = IMDBDataset(data_dir_path=DATA_DIR, filename=TRAIN_FILE, class_names=CLASS_NAMES)
    te_ds = IMDBDataset(data_dir_path=DATA_DIR, filename=TEST_FILE, class_names=CLASS_NAMES)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(CLASS_NAMES)

    # load checkpoint metadata
    ckpt_meta = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt_meta.get("state_dict", {})
    # Lightning may use different keys for hyperparams; try common ones
    hparams = None
    for key in ("hyper_parameters", "hparams", "hyper_parameters_saved", "pytorch-lightning"):
        if key in ckpt_meta:
            if key == "pytorch-lightning":
                hparams = ckpt_meta[key].get("hp", None) or ckpt_meta[key].get("hyper_parameters", None)
            else:
                hparams = ckpt_meta[key]
            if hparams:
                break

    print("Checkpoint hparams (extracted):")
    print(hparams)

    # infer encoder hyperparams (prefer explicit hparams; fallback to state_dict)
    encoder_emb_dim = hparams.get("encoder_emb_dim") if hparams else None
    encoder_n_layers = hparams.get("encoder_n_layers") if hparams else None
    encoder_ff_dim = hparams.get("encoder_ff_dim") if hparams else None
    encoder_n_heads = hparams.get("encoder_n_heads") if hparams else None

    if not encoder_emb_dim or not encoder_ff_dim or not encoder_n_layers:
        token_emb_key = next((k for k in state_dict if "token_emb.weight" in k), None)
        if token_emb_key:
            encoder_emb_dim = state_dict[token_emb_key].shape[1]
        linear1_key = next((k for k in state_dict if "linear1.weight" in k), None)
        if linear1_key:
            encoder_ff_dim = state_dict[linear1_key].shape[0]
        layer_idxs = {
            int(m.group(1))
            for k in state_dict.keys()
            if (m := re.search(r"transformer\.layers\.(\d+)\.", k))
        }
        if layer_idxs:
            encoder_n_layers = max(layer_idxs) + 1
        encoder_n_heads = encoder_n_heads or 4

    encoder_kwargs_ckpt = dict(
        emb_dim=int(encoder_emb_dim or 128),
        n_layers=int(encoder_n_layers or 3),
        n_heads=int(encoder_n_heads or 4),
        ff_dim=int(encoder_ff_dim or (int(encoder_emb_dim or 128) * 2)),
        max_seq_len=256,
    )

    print("Inferred encoder kwargs:", encoder_kwargs_ckpt)

    # get other saved hparams with sensible fallbacks
    num_experts_ckpt = int(hparams.get("num_experts", 8)) if hparams else 8
    expert_hidden_dim_ckpt = int(hparams.get("expert_hidden_dim", 64)) if hparams else 64
    top_k_ckpt = int(hparams.get("top_k", 1)) if hparams else 1

    # build backbone with inferred hyperparameters
    backbone_skeleton = SBERT_MoE_Model(
        num_classes=num_classes,
        num_experts=num_experts_ckpt,
        expert_hidden_dim=expert_hidden_dim_ckpt,
        vocab_texts=tr_ds.texts,
        encoder_kwargs=encoder_kwargs_ckpt,
        top_k=top_k_ckpt,
    )

    print(f"Loading model weights from checkpoint: {CHECKPOINT_PATH}")
    loaded_model = MoE_LightningModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        model=backbone_skeleton,
        num_classes=num_classes,
        num_experts=num_experts_ckpt,
    )

    # device + eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()

    metrics = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
    }).to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    print("\n--- Running Inference on Test Loader (Multi-Class) ---")
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(te_loader):
            labels = labels.to(device)
            logits, _ = loaded_model(texts)
            preds = torch.argmax(logits, dim=1)
            metrics.update(logits, labels)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().tolist())

            if batch_idx < 3:
                probs = torch.softmax(logits, dim=1).cpu()
                for i, (text, pred, p) in enumerate(zip(texts, preds, probs)):
                    actual_lbl = CLASS_NAMES[labels[i].item()]
                    sig_probs = {CLASS_NAMES[j]: round(p[j].item(), 3) for j in range(len(CLASS_NAMES)) if p[j] > 0.1}
                    print(f"\n[Batch {batch_idx} - Sample {i}] Text: {text[:100]}...")
                    print(f"Predicted: {CLASS_NAMES[pred.item()]} | Actual: {actual_lbl}")
                    print(f"High probs: {sig_probs}")

    # final aggregate metrics
    final_results = metrics.compute()
    print("\n" + "=" * 30)
    print("FINAL EVALUATION REPORT (MULTICLASS)")
    print("=" * 30)
    for metric_name, value in final_results.items():
        print(f"{metric_name.capitalize()}: {value.item():.4f}")

    # per-class metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    print("\nClassification Report (per-class):\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    per_class_accuracy = []
    N = len(y_true)
    for i in range(len(CLASS_NAMES)):
        TP = cm[i, i]
        TN = (cm.sum() - cm[i, :].sum() - cm[:, i].sum() + TP)
        per_class_accuracy.append((TP + TN) / N)

    per_class_df = pd.DataFrame({
        "class_idx": list(range(len(CLASS_NAMES))),
        "class_name": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "per_class_accuracy": per_class_accuracy,
    })

    # create output dir structure
    out_root = Path(args.output_dir)
    run_dir = out_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "per_class_metrics.csv"
    per_class_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-class metrics to {csv_path.resolve()}")

    # save classification report to text file
    cls_txt = run_dir / "classification_report.txt"
    with open(cls_txt, "w") as fh:
        fh.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print(f"Saved classification report to {cls_txt.resolve()}")

    # confusion matrix figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix ({args.name})")
    fig.tight_layout()

    cm_path = run_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path.resolve()}")

    # normalized confusion matrix saved as csv for convenience
    cm_norm = (cm.astype(np.float32) / cm.sum(axis=1, keepdims=True))
    cm_norm_path = run_dir / "confusion_matrix_normalized.csv"
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_norm_path)
    print(f"Saved normalized confusion matrix to {cm_norm_path.resolve()}")

    # Save summary JSON (metrics + hparams)
    summary = {
        "run_name": args.name,
        "checkpoint": str(CHECKPOINT_PATH),
        "metrics": {k: float(v.item()) for k, v in final_results.items()},
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "hparams": hparams if hparams else {},
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {summary_path.resolve()}")

    print("\nDone.")
