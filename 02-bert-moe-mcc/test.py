#!/usr/bin/env python
"""
test.py — automatically finds the best local checkpoint (recursively across runs),
reads saved hparams from checkpoint, infers encoder dimensions if needed,
reconstructs model correctly, loads weights, and evaluates metrics (including per-class CSV export).
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
)
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
torch.set_float32_matmul_precision('medium')

# =====================================================
# Helper — recursively find the best checkpoint by val_acc
# =====================================================
def pick_best_local_checkpoint(checkpoints_root="checkpoints/multiclass"):
    ckpt_paths = list(Path(checkpoints_root).rglob("*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_root}")
    best_val, best_path = -1.0, None
    for p in ckpt_paths:
        m = re.search(r"val_acc=([0-9]*\.?[0-9]+)", p.name)
        val = float(m.group(1)) if m else p.stat().st_mtime
        if val > best_val:
            best_val, best_path = val, p
    return best_path

# =====================================================
# Imports — adjust paths if needed
# =====================================================
try:
    from utils.module import MoE_LightningModule, IMDBDataset
    from layers.encoder import SBERT_MoE_Model
    TRAIN_DATA_DIR = Path("../data/imdb_arh_trimmed")
except Exception:
    from module import MoE_LightningModule, IMDBDataset
    from encoder import SBERT_MoE_Model
    TRAIN_DATA_DIR = Path("../data/imdb_arh_trimmed")

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    # 1) find best checkpoint
    CHECKPOINT_PATH = pick_best_local_checkpoint("checkpoints/multiclass")
    print(f"✅ Using best local checkpoint: {CHECKPOINT_PATH}")

    # 2) dataset
    DATA_DIR = TRAIN_DATA_DIR
    TEST_FILE = "imdb_arh_test.csv"
    BATCH_SIZE = 64

    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, "imdb_arh_train.csv")
    print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

    tr_ds = IMDBDataset(data_dir_path=DATA_DIR, filename="imdb_arh_train.csv", class_names=CLASS_NAMES)
    te_ds = IMDBDataset(data_dir_path=DATA_DIR, filename=TEST_FILE, class_names=CLASS_NAMES)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(CLASS_NAMES)

    # 3) Load checkpoint metadata
    ckpt_meta = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt_meta.get("state_dict", {})
    hparams = ckpt_meta.get("hyper_parameters", {})

    print("Checkpoint hparams (extracted):")
    print(hparams)

    # =====================================================
    # Infer encoder hyperparameters from hparams or state_dict
    # =====================================================
    encoder_emb_dim = hparams.get("encoder_emb_dim")
    encoder_n_layers = hparams.get("encoder_n_layers")
    encoder_ff_dim = hparams.get("encoder_ff_dim")
    encoder_n_heads = hparams.get("encoder_n_heads")

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
            if (m := re.search(r"transformer\\.layers\\.(\\d+)\\.", k))
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

    num_experts_ckpt = int(hparams.get("num_experts", 8))
    expert_hidden_dim_ckpt = int(hparams.get("expert_hidden_dim", 128))
    top_k_ckpt = int(hparams.get("top_k", 1))

    # =====================================================
    # Build backbone with matching dimensions
    # =====================================================
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

    # =====================================================
    # Evaluation
    # =====================================================
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

    all_preds, all_labels = [], []

    print("\n--- Running Inference on Test Loader (Multi-Class) ---")
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(te_loader):
            labels = labels.to(device)
            logits, _ = loaded_model(texts)
            preds = torch.argmax(logits, dim=1)
            metrics.update(logits, labels)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if batch_idx < 3:
                probs = torch.softmax(logits, dim=1).cpu()
                for i, (text, pred, p) in enumerate(zip(texts, preds, probs)):
                    actual_lbl = CLASS_NAMES[labels[i].item()]
                    sig_probs = {CLASS_NAMES[j]: round(p[j].item(), 3) for j in range(len(CLASS_NAMES)) if p[j] > 0.1}
                    print(f"\n[Batch {batch_idx} - Sample {i}] Text: {text[:100]}...")
                    print(f"Predicted: {CLASS_NAMES[pred.item()]} | Actual: {actual_lbl}")
                    print(f"High probs: {sig_probs}")

    # =====================================================
    # Final overall metrics
    # =====================================================
    print("\n" + "=" * 30)
    print("FINAL EVALUATION REPORT (MULTICLASS)")
    print("=" * 30)
    results = metrics.compute()
    for metric_name, value in results.items():
        print(f"{metric_name.capitalize()}: {value.item():.4f}")

    # =====================================================
    # Per-class metrics and CSV export
    # =====================================================
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

    print("\nPer-class metrics table:\n")
    print(per_class_df.to_string(index=False))

    csv_path = Path("per_class_metrics.csv")
    per_class_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-class metrics to {csv_path.resolve()}")

    print("\nConfusion Matrix (rows=true labels, cols=predicted labels):")
    print(cm)

    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    print("\nNormalized confusion matrix (per-row):")
    print(np.round(cm_norm, 3))
