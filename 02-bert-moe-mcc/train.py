#!/usr/bin/env python3
"""
train.py — training entrypoint with WandB sweep support and warmup scheduler support.

Each run will save checkpoints to a unique subfolder under checkpoints/multiclass/<run_id>/,
and keep only the best checkpoint for that run (save_top_k=1).
"""
import os
import argparse
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path

import wandb
import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from utils.module import MoE_LightningModule, IMDBDataset
from layers.encoder import SBERT_MoE_Model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run", action="store_true", help="Enable wandb logging (used by sweeps).")
    parser.add_argument("--data_dir", type=str, default="../data/imdb_arh_trimmed", help="Path to data directory")
    parser.add_argument("--train_file", type=str, default="imdb_arh_train.csv", help="Train CSV filename")
    parser.add_argument("--val_file", type=str, default="imdb_arh_val.csv", help="Validation CSV filename")
    parser.add_argument("--project", type=str, default="MovieGenreMulticlassMoE", help="W&B project name")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16, help="Trainer precision: 16 or 32")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--aux_loss_weight", type=float, default=None)
    parser.add_argument("--expert_hidden_dim", type=int, default=None)
    parser.add_argument("--encoder_emb_dim", type=int, default=None)
    parser.add_argument("--warmup_frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    # --------- Configuration ----------
    
    args = parse_args()

    # If running under a sweep, wandb.agent will call this script and set wandb.config.
    run = None
    if args.wandb_run and ("WANDB_API_KEY" in os.environ):
        run = wandb.init(project=args.project, reinit=True)
        config = run.config
    else:
        default_cfg = dict(
            seed=42,
            learning_rate=3e-4,
            batch_size=64,
            num_experts=8,
            top_k=2,
            aux_loss_weight=0.01,
            expert_hidden_dim=128,
            encoder_emb_dim=256,
            warmup_frac=0.1,
            max_epochs=args.max_epochs,
        )
        # override from CLI if set
        for k in ["learning_rate", "batch_size", "num_experts", "top_k", "aux_loss_weight",
                  "expert_hidden_dim", "encoder_emb_dim", "warmup_frac", "seed", "max_epochs"]:
            val = getattr(args, k, None)
            if val is not None:
                default_cfg[k] = val
        config = default_cfg

    # Unpack config
    seed = int(config.get("seed", 42))
    learning_rate = float(config.get("learning_rate", 3e-4))
    batch_size = int(config.get("batch_size", 64))
    num_experts = int(config.get("num_experts", 8))
    top_k = int(config.get("top_k", 2))
    aux_loss_weight = float(config.get("aux_loss_weight", 0.01))
    expert_hidden_dim = int(config.get("expert_hidden_dim", 128))
    encoder_emb_dim = int(config.get("encoder_emb_dim", 256))
    warmup_frac = float(config.get("warmup_frac", 0.1))
    max_epochs = int(config.get("max_epochs", args.max_epochs))

    seed_everything(seed)

    # Fetch data and discover classes
    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Train/Val files not found under data_dir")

    CLASS_NAMES = IMDBDataset.discover_classes(data_dir, args.train_file)
    num_classes = len(CLASS_NAMES)
    print(f"Discovered {num_classes} classes: {CLASS_NAMES}")

    # Build vocab texts and label weights for tokenizer
    train_df = pd.read_csv(train_path)
    vocab_texts = train_df["description"].fillna("").astype(str).tolist()
    class_counts = train_df['csv_genre'].value_counts().sort_index()
    weights = 1.0 / (class_counts / class_counts.sum())  # inverse frequency
    weights = (weights / weights.sum()).tolist()

    # --------- Model ----------

    # Prepare model
    encoder_kwargs = dict(
        emb_dim=encoder_emb_dim,
        n_layers=3,
        n_heads=4,
        ff_dim=encoder_emb_dim * 2,
        max_seq_len=256,
    )

    backbone = SBERT_MoE_Model(
        num_classes=num_classes,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden_dim,
        top_k=top_k,
        vocab_texts=vocab_texts,
        encoder_kwargs=encoder_kwargs,
    )

    pl_module = MoE_LightningModule(
        model=backbone,
        num_classes=num_classes,
        num_experts=num_experts,
        learning_rate=learning_rate,
        aux_loss_weight=aux_loss_weight,
        class_weights=weights,
    )

    # Save hyperparameters to pl_module for checkpointing
    def _safe_set_hparam(module, key, val):
        try:
            module.hparams[key] = val
        except Exception:
            setattr(module, key, val)

    if run is not None:
        try:
            for k, v in dict(run.config).items():
                _safe_set_hparam(pl_module, k, v)
        except Exception:
            setattr(pl_module, "wandb_config", dict(run.config))

    _arch_hparams = {
        "num_experts": num_experts,
        "expert_hidden_dim": expert_hidden_dim,
        "top_k": top_k,
        "encoder_emb_dim": encoder_emb_dim,
        "encoder_n_layers": encoder_kwargs.get("n_layers", 3),
        "encoder_n_heads": encoder_kwargs.get("n_heads", 4),
        "encoder_ff_dim": encoder_kwargs.get("ff_dim", encoder_emb_dim * 2),
        "aux_loss_weight": aux_loss_weight,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "warmup_frac": warmup_frac,
        "max_epochs": max_epochs,
        "class_weights": weights if 'weights' in locals() else None,
    }

    for k, v in _arch_hparams.items():
        if v is not None:
            _safe_set_hparam(pl_module, k, v)

    # Datasets and dataloaders
    tr_ds = IMDBDataset(data_dir_path=data_dir, filename=args.train_file, class_names=CLASS_NAMES)
    va_ds = IMDBDataset(data_dir_path=data_dir, filename=args.val_file, class_names=CLASS_NAMES)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)

    # --------- callbacks & logging ----------

    # Early stopping callback
    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="max"
    )

    # Checkpointing callback
    run_id = None
    if run is not None:
        # W&B run id is unique; sanitize for filesystem
        run_id = str(run.id).replace("/", "-")
    else:
        run_id = f"local-{int(time.time())}"
    ckpt_dir = Path("checkpoints") / "multiclass" / run_id

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_id}-moe-{{epoch:02d}}-{{val_acc:.3f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,    # keep only the best checkpoint for this run
        save_last=False,
    )

    # WandB logger (log_model=True allows retrieving model artifact from W&B if needed)
    wandb_logger = WandbLogger(project=config.get("project", args.project), name=None, log_model=True) if (run is not None or args.wandb_run) else None

    # --------- Training ----------

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[early_stop_cb, checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
    )

    print("Starting training with config:", {
        "seed": seed,
        "lr": learning_rate,
        "batch_size": batch_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "aux_loss_weight": aux_loss_weight,
        "expert_hidden_dim": expert_hidden_dim,
        "encoder_emb_dim": encoder_emb_dim,
        "warmup_frac": warmup_frac,
        "max_epochs": max_epochs,
        "ckpt_dir": str(ckpt_dir),
    })

    trainer.fit(pl_module, train_dataloaders=tr_loader, val_dataloaders=va_loader)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
