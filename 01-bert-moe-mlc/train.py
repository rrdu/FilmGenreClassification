#!/usr/bin/env python3
# train.py
"""
Multilabel training script.
Saves best checkpoint under checkpoints/multilabel/<run_id>/ and logs to wandb if requested.
"""
import os
import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure these match your actual file structure
from utils.module import MoE_LightningModule, IMDBDataset
from layers.moe import SBERT_MoE_Model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run", action="store_true")
    parser.add_argument("--data_dir", type=str, default="../data/imdb_arh_trimmed")
    parser.add_argument("--train_file", type=str, default="imdb_arh_train.csv")
    parser.add_argument("--val_file", type=str, default="imdb_arh_val.csv")
    parser.add_argument("--project", type=str, default="MovieGenreMultilabelMoE")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--expert_hidden_dim", type=int, default=128)
    parser.add_argument("--encoder_emb_dim", type=int, default=256)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pos_weight", action="store_true", help="Disable class balancing weights")
    parser.add_argument("--finetune_backbone", type=int, default=0, help="0=Freeze SBERT, 1=Finetune SBERT")
    parser.add_argument("--threshold", type=float, default=0.3, help="Decision threshold for multilabel (Default: 0.3)")
    
    return parser.parse_args()


def compute_pos_weight(train_df, class_names, label_col="genre_list"):
    """
    Compute pos_weight vector (num_classes,) for BCEWithLogitsLoss.
    Splits by comma to handle "Drama, Romance".
    """
    counts = np.zeros(len(class_names), dtype=np.int64)
    for s in train_df[label_col].fillna("").astype(str):
        # Split by comma and strip whitespace
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if p in class_names:
                counts[class_names.index(p)] += 1
    N = len(train_df)
    pos = counts
    neg = N - counts
    pos_weight = np.ones(len(class_names), dtype=np.float32)
    for i in range(len(class_names)):
        if pos[i] > 0:
            pos_weight[i] = neg[i] / (pos[i] + 1e-8)
        else:
            pos_weight[i] = 1.0
    return torch.tensor(pos_weight, dtype=torch.float32)


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file
    assert train_path.exists() and val_path.exists(), "Train/Val CSVs not found"

    # discover classes (using comma separation inside discover_classes)
    CLASS_NAMES = IMDBDataset.discover_classes(data_dir, args.train_file, label_col='genre_list')
    num_classes = len(CLASS_NAMES)
    print("Discovered classes:", CLASS_NAMES)

    # Read train dataframe for weight calculation
    train_df = pd.read_csv(train_path)

    # pos_weight for BCEWithLogitsLoss (using comma separator)
    if args.no_pos_weight:
        print("⚖️ Class balancing (pos_weight) is DISABLED.")
        pos_weight_list = None
    else:
        # Compute weights as before
        pos_weight_tensor = compute_pos_weight(train_df, CLASS_NAMES, label_col='genre_list')
        print("⚖️ Class balancing (pos_weight) is ENABLED.")
        print("Sample weights:", pos_weight_tensor[:10])
        pos_weight_list = pos_weight_tensor.tolist()

    # Construct backbone
    backbone = SBERT_MoE_Model(
        num_classes=num_classes,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        top_k=args.top_k
    )

    # -------------------------------------------------------------------------
    # Freezing / Unfreezing Logic
    # -------------------------------------------------------------------------
    if args.finetune_backbone == 1:
        print(f"🔥 Finetuning Backbone ENABLED. Threshold: {args.threshold}")
        # Enable gradients for SBERT
        for param in backbone.sbert.parameters():
            param.requires_grad = True
    else:
        print(f"❄️ Finetuning Backbone DISABLED (Frozen). Threshold: {args.threshold}")
        # Freeze SBERT
        for param in backbone.sbert.parameters():
            param.requires_grad = False
    # -------------------------------------------------------------------------

    # Create LightningModule
    pl_module = MoE_LightningModule(
        model=backbone,
        num_classes=num_classes,
        num_experts=args.num_experts,
        learning_rate=args.learning_rate,
        aux_loss_weight=args.aux_loss_weight,
        pos_weight=pos_weight_list,
        multilabel=True,
        # Pass the new arguments
        threshold=args.threshold,
        finetune_backbone=bool(args.finetune_backbone)
    )

    # Save hparams to module
    try:
        for k, v in vars(args).items():
            pl_module.hparams[k] = v
        pl_module.hparams.update({
            "pos_weight": pos_weight_list,
        })
    except Exception:
        pass

    # Datasets and dataloaders
    tr_ds = IMDBDataset(data_dir_path=data_dir, filename=args.train_file, class_names=CLASS_NAMES, label_col='genre_list', multilabel=True)
    va_ds = IMDBDataset(data_dir_path=data_dir, filename=args.val_file, class_names=CLASS_NAMES, label_col='genre_list', multilabel=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # callbacks & logging
    run = None
    wandb_logger = None
    if args.wandb_run and ("WANDB_API_KEY" in os.environ):
        run = wandb.init(project=args.project, reinit=True)
        wandb_logger = WandbLogger(project=args.project, log_model=True)

    run_id = str(run.id).replace("/", "-") if run is not None else f"local-{int(time.time())}"
    ckpt_dir = Path("checkpoints") / "multilabel" / run_id
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_id}-moe-{{epoch:02d}}-{{val_loss:.3f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False)
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        callbacks=[early_stop_cb, checkpoint_cb],
        logger=wandb_logger,
        precision=args.precision)

    trainer.fit(pl_module, train_dataloaders=tr_loader, val_dataloaders=va_loader)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()