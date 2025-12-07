#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py - find best checkpoint and run multilabel evaluation.

Key Features:
- Automatically picks best checkpoint from local directory.
- Computes per-class metrics and confusion matrices.
- Generates advanced visualizations (ROC, PR Curves, Heatmaps, Jaccard).
- Calculates Subset Accuracy (Exact Match).
- Calculates Top-K Metrics (Precision@K, Recall@K, Jaccard@K).

Usage:
    python test.py --data_dir ../data/imdb_arh_trimmed --test_file imdb_arh_test.csv
"""
import os
import argparse
import re
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve, 
    auc, 
    average_precision_score, 
    precision_recall_curve, 
    f1_score, 
    jaccard_score,
    accuracy_score
)

from utils.module import MoE_LightningModule, IMDBDataset
from layers.moe import SBERT_MoE_Model


def pick_best_local_checkpoint(checkpoints_root="checkpoints/multilabel"):
    root = Path(checkpoints_root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_root}")
    
    ckpt_paths = list(root.rglob("*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_root}")
    
    best_val = None
    best_path = None
    for p in ckpt_paths:
        # Try to find val_loss in filename
        m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", p.name)
        if m:
            val = float(m.group(1))
            if best_val is None or val < best_val:
                best_val = val
                best_path = p
        else:
            # Fallback: pick most recent
            if best_path is None or p.stat().st_mtime > best_path.stat().st_mtime:
                best_path = p
                
    return str(best_path)


def calculate_metrics_at_k(y_true, y_prob, k):
    """
    Computes Precision@k, Recall@k, and Jaccard@k (IoU) for multi-label data.
    """
    # Get indices of top k probabilities for each sample
    # argsort sorts ascending, so we take the last k and reverse them
    top_k_idx = np.argsort(y_prob, axis=1)[:, -k:][:, ::-1]
    
    # Create a binary prediction matrix for top k
    y_pred_k = np.zeros_like(y_true, dtype=int)
    for i in range(len(y_true)):
        y_pred_k[i, top_k_idx[i]] = 1
        
    # Intersection (True Positives)
    tp = np.sum(y_true * y_pred_k, axis=1)
    
    # 1. Precision@k = TP / k
    precision_at_k = tp / k
    
    # 2. Recall@k = TP / (Total True Labels)
    true_counts = np.sum(y_true, axis=1)
    recall_at_k = np.divide(tp, true_counts, out=np.zeros_like(tp, dtype=float), where=true_counts!=0)
    
    # 3. Jaccard@k = TP / (Total True + k - TP)
    # Union of indices = (set of true) U (set of predicted k)
    # Size(Union) = Size(True) + Size(Pred_k) - Size(Intersection)
    # Size(Pred_k) is always k
    union_counts = true_counts + k - tp
    jaccard_at_k = np.divide(tp, union_counts, out=np.zeros_like(tp, dtype=float), where=union_counts!=0)
    
    return precision_at_k.mean(), recall_at_k.mean(), jaccard_at_k.mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=f"test_{int(time.time())}")
    parser.add_argument("--data_dir", type=str, default="../data/imdb_arh_trimmed")
    parser.add_argument("--train_file", type=str, default="imdb_arh_train.csv")
    parser.add_argument("--test_file", type=str, default="imdb_arh_test.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="sigmoid threshold for deciding positives")
    return parser.parse_args()


def main():
    args = parse_args()

    CHECKPOINT_PATH = args.checkpoint_path or pick_best_local_checkpoint("checkpoints/multilabel")
    print("Using checkpoint:", CHECKPOINT_PATH)

    DATA_DIR = Path(args.data_dir)
    TRAIN_FILE = args.train_file
    TEST_FILE = args.test_file
    BATCH_SIZE = args.batch_size

    # Discover classes
    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, TRAIN_FILE)
    print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

    te_ds = IMDBDataset(data_dir_path=DATA_DIR, filename=TEST_FILE, class_names=CLASS_NAMES, multilabel=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(CLASS_NAMES)

    # Load checkpoint metadata
    ckpt_meta = torch.load(CHECKPOINT_PATH, map_location="cpu")
    hparams = None
    
    # Robust hparam extraction
    for key in ("hyper_parameters", "hparams", "hyper_parameters_saved", "pytorch-lightning"):
        if key in ckpt_meta:
            if key == "pytorch-lightning":
                hparams = ckpt_meta[key].get("hp", None) or ckpt_meta[key].get("hyper_parameters", None)
            else:
                hparams = ckpt_meta[key]
            if hparams:
                break
    
    print("Checkpoint hparams (extracted):", hparams)

    # Defaults updated to 128 to match notebook
    num_experts_ckpt = int(hparams.get("num_experts", 8)) if hparams else 8
    expert_hidden_dim_ckpt = int(hparams.get("expert_hidden_dim", 128)) if hparams else 128 
    top_k_ckpt = int(hparams.get("top_k", 2)) if hparams else 2

    # Initialize Skeleton
    backbone_skeleton = SBERT_MoE_Model(
        num_classes=num_classes,
        num_experts=num_experts_ckpt,
        expert_hidden_dim=expert_hidden_dim_ckpt,
        top_k=top_k_ckpt,
    )

    print(f"Loading weights with config: Experts={num_experts_ckpt}, Hidden={expert_hidden_dim_ckpt}, TopK={top_k_ckpt}")
    
    # Load Weights
    loaded_model = MoE_LightningModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        model=backbone_skeleton,
        num_classes=num_classes,
        num_experts=num_experts_ckpt,
        multilabel=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(te_loader):
            labels = labels.to(device).float()
            logits, _ = loaded_model(texts)
            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)      # shape (N, C)
    y_pred = np.array(all_preds)       # shape (N, C)
    y_prob = np.array(all_probs)

    # --- Create Output Directory ---
    out_root = Path(args.output_dir)
    run_dir = out_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = run_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir.resolve()}")

    # =========================================================
    # VISUALIZATIONS
    # =========================================================
    print("Generating advanced visualizations...")
    
    # 1) ROC & PR per-class + micro
    auc_per = []
    ap_per = []
    for i, cname in enumerate(CLASS_NAMES):
        y_true_i = y_true[:, i]
        y_score_i = y_prob[:, i]
        try:
            fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
            roc_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr, roc_auc = [0.0], [0.0], 0.0
        try:
            ap = average_precision_score(y_true_i, y_score_i)
        except Exception:
            ap = 0.0
        auc_per.append(float(roc_auc))
        ap_per.append(float(ap))

    # micro-average ROC & AP
    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_prob.ravel())
        auc_micro = float(auc(fpr_micro, tpr_micro))
    except Exception:
        auc_micro = 0.0

    # Plot micro ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr_micro, tpr_micro, label=f"micro ROC (AUC={auc_micro:.3f})")
    plt.plot([0,1], [0,1], 'k--', linewidth=0.6)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Micro-average ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(visual_dir / "roc_micro.png", dpi=200)
    plt.close()

    # Save per-class AUC/AP
    pd.DataFrame({
        "class": CLASS_NAMES,
        "roc_auc": auc_per,
        "avg_precision": ap_per
    }).to_csv(visual_dir / "per_class_auc_ap.csv", index=False)

    # 2) Precision-Recall curves for worst classes by AP
    k_plot = min(12, len(CLASS_NAMES))
    order = np.argsort(ap_per)[:k_plot]  # worst by AP
    plt.figure(figsize=(10,8))
    for idx in order:
        p, r, _ = precision_recall_curve(y_true[:, idx], y_prob[:, idx])
        plt.plot(r, p, label=f"{CLASS_NAMES[idx]} (AP={ap_per[idx]:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (worst classes)")
    plt.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig(visual_dir / "pr_worst_classes.png", dpi=200)
    plt.close()

    # 3) Threshold sweep (micro F1)
    threshs = np.linspace(0.1, 0.9, 17)
    micro_f1s = []
    for t in threshs:
        preds_t = (y_prob >= t).astype(int)
        micro_f1s.append(float(f1_score(y_true, preds_t, average='micro', zero_division=0)))

    plt.figure(figsize=(6,4))
    plt.plot(threshs, micro_f1s, '-o')
    plt.xlabel("threshold"); plt.ylabel("micro F1"); plt.title("Micro F1 vs threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(visual_dir / "micro_f1_vs_threshold.png", dpi=200)
    plt.close()

    best_idx = int(np.argmax(micro_f1s))
    best_thresh = float(threshs[best_idx])
    with open(run_dir / "best_threshold.txt", "w") as fh:
        fh.write(f"{best_thresh:.4f}\n")
    print(f"Best micro-F1 {micro_f1s[best_idx]:.4f} at threshold {best_thresh:.2f}")

    # 4) Label co-occurrence heatmap (normalized by true-label counts)
    cooccur = (y_true.T @ y_true).astype(int)
    cooccur_norm = cooccur / (cooccur.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(10,8))
    sns.heatmap(cooccur_norm, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.title("Label co-occurrence (normalized by true-label counts)")
    plt.tight_layout()
    plt.savefig(visual_dir / "label_cooccurrence.png", dpi=200)
    plt.close()

    # 5) Pairwise predicted-vs-true heatmap P(pred=j | true=i)
    pairwise = np.zeros((num_classes, num_classes), dtype=int)
    for n in range(y_true.shape[0]):
        true_idx = np.where(y_true[n] == 1)[0]
        pred_idx = np.where(y_pred[n] == 1)[0]
        for i in true_idx:
            for j in pred_idx:
                pairwise[i, j] += 1
    pairwise_norm = pairwise / (pairwise.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(10,8))
    sns.heatmap(pairwise_norm, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Reds")
    plt.xlabel("predicted label"); plt.ylabel("true label")
    plt.title("P(pred=j | true=i)")
    plt.tight_layout()
    plt.savefig(visual_dir / "pairwise_pred_vs_true.png", dpi=200)
    plt.close()

    # 6) Per-class Jaccard (IoU) and barplot
    jaccard_per = []
    for i in range(num_classes):
        try:
            j = jaccard_score(y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
        except Exception:
            j = 0.0
        jaccard_per.append(float(j))
    pd.DataFrame({"class": CLASS_NAMES, "jaccard": jaccard_per}).to_csv(visual_dir / "per_class_jaccard.csv", index=False)

    plt.figure(figsize=(8, max(4, num_classes*0.12)))
    plt.barh(CLASS_NAMES, jaccard_per)
    plt.xlabel("Jaccard (IoU)")
    plt.title("Per-class Jaccard similarity")
    plt.tight_layout()
    plt.savefig(visual_dir / "per_class_jaccard.png", dpi=200)
    plt.close()

    # =========================================================
    # STANDARD METRICS & CONFUSION MATRICES
    # =========================================================
    
    # Calculate Subset Accuracy (Exact Match)
    subset_acc = accuracy_score(y_true, y_pred)
    
    # Calculate Top-K Metrics
    k_values = [1, 3, 5, 10]
    top_k_metrics = {}
    print("\n--- Top-K Metrics ---")
    for k in k_values:
        if k > num_classes: continue
        pk, rk, jk = calculate_metrics_at_k(y_true, y_prob, k)
        top_k_metrics[f"P@{k}"] = pk
        top_k_metrics[f"R@{k}"] = rk
        top_k_metrics[f"Jaccard@{k}"] = jk
        print(f"k={k}: Precision={pk:.4f}, Recall={rk:.4f}, Jaccard={jk:.4f}")

    # Macro / micro average metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    print("\n--- Summary Metrics ---")
    print(f"Subset Accuracy (Exact Match): {subset_acc:.4f}")
    print("Macro Precision {:.4f} Recall {:.4f} F1 {:.4f}".format(precision_macro, recall_macro, f1_macro))
    print("Micro Precision {:.4f} Recall {:.4f} F1 {:.4f}".format(precision_micro, recall_micro, f1_micro))

    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_df = pd.DataFrame({
        "class_idx": list(range(num_classes)),
        "class_name": CLASS_NAMES,
        "precision": precision_per,
        "recall": recall_per,
        "f1": f1_per,
        "support": support_per
    })

    # confusion matrices: multilabel_confusion_matrix returns (C, 2, 2)
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    csv_path = run_dir / "per_class_metrics.csv"
    per_class_df.to_csv(csv_path, index=False)
    print(f"Saved per-class metrics to {csv_path.resolve()}")

    # classification report
    cls_txt = run_dir / "classification_report.txt"
    with open(cls_txt, "w") as fh:
        fh.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print(f"Saved classification report to {cls_txt.resolve()}")

    # Plot confusion matrices: arrange subplots in grid
    C = num_classes
    cols = min(4, C)
    rows = (C + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)  # flatten
    for i in range(C):
        ax = axes[i]
        disp = ConfusionMatrixDisplay(confusion_matrix=mcm[i], display_labels=[f"not {CLASS_NAMES[i]}", CLASS_NAMES[i]])
        disp.plot(ax=ax, values_format='d', cmap=plt.cm.Blues)
        ax.set_title(CLASS_NAMES[i])
    # hide unused axes
    for j in range(C, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    cm_path = run_dir / "confusion_matrices_per_class.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved per-class confusion matrices to {cm_path.resolve()}")

    # save normalized confusion matrices CSVs (per-class)
    norm_dir = run_dir / "confusion_matrices_csv"
    norm_dir.mkdir(parents=True, exist_ok=True)
    for i in range(C):
        cm_i = mcm[i]
        # For 2x2, normalize by sum
        cm_norm = cm_i.astype(float) / (cm_i.sum() + 1e-8)
        pd.DataFrame(cm_norm, index=["tn","fp"], columns=["fn","tp"]).to_csv(norm_dir / f"{i}_{CLASS_NAMES[i]}_cm_norm.csv")
    print(f"Saved normalized 2x2 confusion matrices CSVs to {norm_dir.resolve()}")

    # Save summary
    summary = {
        "run_name": args.name,
        "checkpoint": str(CHECKPOINT_PATH),
        "metrics": {
            "subset_accuracy": float(subset_acc),
            "top_k_metrics": top_k_metrics,
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_micro": float(precision_micro),
            "recall_micro": float(recall_micro),
            "f1_micro": float(f1_micro),
        },
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "hparams": hparams if hparams else {}
    }
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {run_dir / 'summary.json'}")

    print("Done.")

if __name__ == "__main__":
    main()