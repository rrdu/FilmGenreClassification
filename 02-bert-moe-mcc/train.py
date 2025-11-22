#!/usr/bin/env python
"""
train.py — training entrypoint with WandB sweep support and warmup scheduler support.

Usage:
  # manual run (no wandb)
  python /mnt/data/train.py

  # manual run but log to wandb (useful when debugging)
  export WANDB_API_KEY=<key>
  python /mnt/data/train.py --wandb_run

  # When using wandb sweeps, start agents as described in docs; wandb.agent will run this script.
"""
import os
import argparse
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Try multiple import locations (adapt if your project structure differs)
try:
    from utils.module import MoE_LightningModule, IMDBDataset
    from layers.encoder import SBERT_MoE_Model
except Exception:
    try:
        from module import MoE_LightningModule, IMDBDataset
        from encoder import SBERT_MoE_Model
    except Exception:
        from multiclass_tr import MoE_LightningModule, IMDBDataset, SBERT_MoE_Model

import wandb


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
    # allow overriding some defaults from CLI when debugging locally
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
    args = parse_args()

    # If running under a sweep, wandb.agent will call this script and set wandb.config.
    run = None
    if args.wandb_run and ("WANDB_API_KEY" in os.environ):
        run = wandb.init(project=args.project, reinit=True)
        config = run.config
    else:
        # Default config for manual runs; these are used if no wandb config is present
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
        # Override defaults by CLI args if provided
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

    # --------- data and classes ----------
    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path.resolve()}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path.resolve()}")

    # Discover classes using the dataset helper
    CLASS_NAMES = IMDBDataset.discover_classes(data_dir, args.train_file)
    num_classes = len(CLASS_NAMES)
    print(f"Discovered {num_classes} classes: {CLASS_NAMES}")

    # Build vocab texts from the whole training CSV for tokenizer
    train_df = pd.read_csv(train_path)
    vocab_texts = train_df["description"].fillna("").astype(str).tolist()

    # --------- model construction ----------
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
    )

    # Ensure warmup_frac is available for configure_optimizers
    try:
        pl_module.hparams["warmup_frac"] = warmup_frac
    except Exception:
        setattr(pl_module, "warmup_frac", warmup_frac)

    # If running under wandb, merge wandb.config into module hparams so they get saved to ckpt
    if run is not None:
        # wandb.config is a dict-like of the current run's hyperparams
        try:
            for k, v in dict(run.config).items():
                # store in hparams for Lightning checkpoint
                pl_module.hparams[k] = v
        except Exception:
            # safe fallback: set attribute
            setattr(pl_module, "wandb_config", dict(run.config))

    # --------- datasets & dataloaders ----------
    tr_ds = IMDBDataset(data_dir_path=data_dir, filename=args.train_file, class_names=CLASS_NAMES)
    va_ds = IMDBDataset(data_dir_path=data_dir, filename=args.val_file, class_names=CLASS_NAMES)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)

    # --------- callbacks & logging ----------
    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="max"
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/multiclass/",
        filename="moe-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True
    )

    # WandB logger
    if run is not None or args.wandb_run:
        # log_model=True saves model file to W&B run so it can be downloaded later if needed
        wandb_logger = WandbLogger(project=config.get("project", args.project), name=None, log_model=True)
    else:
        wandb_logger = None  # no logger

    # --------- trainer ----------
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
    })

    trainer.fit(pl_module, train_dataloaders=tr_loader, val_dataloaders=va_loader)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
