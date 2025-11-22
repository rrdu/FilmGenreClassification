#!/usr/bin/env python
"""
test.py — picks the best local checkpoint by val_acc, rebuilds the exact model
from saved hyperparameters (if present), loads weights, and runs evaluation.
Run:
  python /mnt/data/test.py
"""
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy
)
import glob
import re
from pathlib import Path
import json

# Helpers to find best local checkpoint
def pick_best_local_checkpoint(checkpoints_dir="checkpoints/multiclass"):
    ckpt_paths = glob.glob(str(Path(checkpoints_dir) / "*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    best_val = -1.0
    best_path = None
    for p in ckpt_paths:
        m = re.search(r"val_acc=([0-9]*\.?[0-9]+)", p)
        if m:
            val = float(m.group(1))
        else:
            # fallback: use file modification time (older runs < newer runs)
            val = Path(p).stat().st_mtime
        if val > best_val:
            best_val = val
            best_path = p
    if best_path is None:
        raise RuntimeError("Couldn't determine best checkpoint")
    return best_path

# Project imports (adjust if necessary)
try:
    from utils.module import MoE_LightningModule, IMDBDataset
    from layers.encoder import SBERT_MoE_Model
    # optional: import default train-time constants if they exist to use as fallback
    from train import k as TRAIN_K, num_experts as TRAIN_NUM_EXPERTS, expert_hidden_dim as TRAIN_EXPERT_HDIM, encoder_kwargs as TRAIN_ENCODER_KWARGS, DATA_DIR as TRAIN_DATA_DIR
except Exception:
    # best-effort fallback names if train.py layout is different
    TRAIN_K = 1
    TRAIN_NUM_EXPERTS = 8
    TRAIN_EXPERT_HDIM = 128
    TRAIN_ENCODER_KWARGS = dict(emb_dim=256, n_layers=3, n_heads=4, ff_dim=512, max_seq_len=256)
    TRAIN_DATA_DIR = Path("../data/imdb_arh_trimmed")

if __name__ == "__main__":
    # 1) find best checkpoint
    try:
        CHECKPOINT_PATH = pick_best_local_checkpoint("checkpoints/multiclass")
        print(f"✅ Using best local checkpoint: {CHECKPOINT_PATH}")
    except Exception as e:
        raise RuntimeError(f"No valid checkpoint found: {e}")

    # 2) discover classes and load datasets
    DATA_DIR = TRAIN_DATA_DIR
    TEST_FILE = "imdb_arh_test.csv"
    BATCH_SIZE = 64

    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, "imdb_arh_train.csv")
    print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

    tr_ds = IMDBDataset(data_dir_path=DATA_DIR, filename="imdb_arh_train.csv", class_names=CLASS_NAMES)
    te_ds = IMDBDataset(data_dir_path=DATA_DIR, filename=TEST_FILE, class_names=CLASS_NAMES)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(CLASS_NAMES)

    # 3) inspect checkpoint metadata for saved hyperparameters
    ckpt_meta = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # Lightning may store hparams under several keys; try common ones
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

    # 4) derive model creation args from saved hparams (fallback to train defaults)
    def _int(h, k, default):
        try:
            return int(h.get(k)) if h and k in h else int(default)
        except Exception:
            return int(default)

    def _float(h, k, default):
        try:
            return float(h.get(k)) if h and k in h else float(default)
        except Exception:
            return float(default)

    num_experts_ckpt = _int(hparams, "num_experts", TRAIN_NUM_EXPERTS)
    expert_hidden_dim_ckpt = _int(hparams, "expert_hidden_dim", TRAIN_EXPERT_HDIM)
    top_k_ckpt = _int(hparams, "top_k", TRAIN_K)

    # Attempt to capture encoder kwargs that may have been saved under readable names
    encoder_emb_dim = _int(hparams, "encoder_emb_dim", TRAIN_ENCODER_KWARGS.get("emb_dim", 256))
    encoder_n_layers = _int(hparams, "encoder_n_layers", TRAIN_ENCODER_KWARGS.get("n_layers", 3))
    encoder_n_heads = _int(hparams, "encoder_n_heads", TRAIN_ENCODER_KWARGS.get("n_heads", 4))
    encoder_ff_dim = _int(hparams, "encoder_ff_dim", TRAIN_ENCODER_KWARGS.get("ff_dim", encoder_emb_dim * 2))
    encoder_max_seq_len = _int(hparams, "encoder_max_seq_len", TRAIN_ENCODER_KWARGS.get("max_seq_len", 256))

    encoder_kwargs_ckpt = dict(
        emb_dim=encoder_emb_dim,
        n_layers=encoder_n_layers,
        n_heads=encoder_n_heads,
        ff_dim=encoder_ff_dim,
        max_seq_len=encoder_max_seq_len,
    )

    print("Reconstructed model config for loading:")
    print({
        "num_experts": num_experts_ckpt,
        "expert_hidden_dim": expert_hidden_dim_ckpt,
        "top_k": top_k_ckpt,
        "encoder_kwargs": encoder_kwargs_ckpt,
    })

    # 5) instantiate backbone with the derived hyperparameters
    backbone_skeleton = SBERT_MoE_Model(
        num_classes=num_classes,
        num_experts=num_experts_ckpt,
        expert_hidden_dim=expert_hidden_dim_ckpt,
        vocab_texts=tr_ds.texts,
        encoder_kwargs=encoder_kwargs_ckpt,
        top_k=top_k_ckpt,
    )

    # 6) load checkpoint weights into Lightning module
    print(f"Loading model weights from checkpoint: {CHECKPOINT_PATH}")
    loaded_model = MoE_LightningModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        model=backbone_skeleton,
        num_classes=num_classes,
        num_experts=num_experts_ckpt,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()

    # 7) metrics (same as your evaluation pipeline)
    metrics = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
    }).to(device)

    def map_logits_to_labels_multiclass(logits, class_names):
        probs = torch.softmax(logits, dim=1).cpu()
        pred_indices = torch.argmax(probs, dim=1)
        preds = [class_names[idx.item()] for idx in pred_indices]
        return preds, probs

    # 8) inference loop
    print("\n--- Running Inference on Test Loader (Multi-Class) ---")

    with torch.no_grad():
        for batch_idx, batch in enumerate(te_loader):
            texts, true_label_indices = batch
            true_label_indices = true_label_indices.to(device)

            logits, _ = loaded_model(texts)
            metrics.update(logits, true_label_indices)

            if batch_idx < 3:
                preds, probs = map_logits_to_labels_multiclass(logits, CLASS_NAMES)
                for i, (text, pred, p) in enumerate(zip(texts, preds, probs)):
                    actual_lbl = CLASS_NAMES[true_label_indices[i].item()]
                    sig_probs = {CLASS_NAMES[j]: round(p[j].item(), 3) for j in range(len(CLASS_NAMES)) if p[j] > 0.1}
                    print(f"\n[Batch {batch_idx} - Sample {i}]")
                    print(f"Text: {text[:100]}...")
                    print(f"Predicted: {pred}")
                    print(f"Actual: {actual_lbl}")
                    print(f"High probs: {sig_probs}")

    # 9) final metrics
    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT (MULTICLASS)")
    print("="*30)

    final_results = metrics.compute()
    for metric_name, value in final_results.items():
        print(f"{metric_name.capitalize()}: {value.item():.4f}")

    metrics.reset()
