import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl

from layers.encoder import SBERT_MoE_Model


from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dotenv import load_dotenv
import wandb

from pathlib import Path
from torch.utils.data import DataLoader
from utils.module import MoE_LightningModule
from utils.module import IMDBDataset

# hyperparams used in main
k = 2
num_experts = 5
expert_hidden_dim = 64
DATA_DIR = Path('../data/imdb_arh_trimmed')
NUM_WORKERS = 4
    
if __name__ == "__main__":    
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, 'imdb_arh_train.csv')
    print(f"Discovered classes: {CLASS_NAMES}")
    
    tr_ds = IMDBDataset(data_dir_path=DATA_DIR, filename='imdb_arh_train.csv', class_names=CLASS_NAMES)
    va_ds = IMDBDataset(data_dir_path=DATA_DIR, filename='imdb_arh_val.csv', class_names=CLASS_NAMES)

    tr_loader = DataLoader(tr_ds, batch_size=32, num_workers=NUM_WORKERS, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, num_workers=NUM_WORKERS, shuffle=False)

    wandb_logger = WandbLogger(
        project="MovieGenreMulticlassMoE", 
        name=f"{num_experts}-experts-{k}-topk",
        log_model=False,
    )
    wandb_logger.experiment.config.update({"class_names": CLASS_NAMES})

    num_classes = len(CLASS_NAMES)
    encoder_kwargs = dict(
        emb_dim=256,
        n_layers=4,
        n_heads=8,
        ff_dim=512,
        max_seq_len=256,
    )
    backbone = SBERT_MoE_Model(
        num_classes=num_classes, 
        num_experts=num_experts, 
        expert_hidden_dim=expert_hidden_dim,
        vocab_texts=tr_ds.texts,
        encoder_kwargs=encoder_kwargs,
        top_k=k,
    )

    pl_module = MoE_LightningModule(
        model=backbone, 
        num_classes=num_classes, 
        num_experts=num_experts,
        learning_rate=3e-4,
        aux_loss_weight=0.1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min"
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/multiclass/",
        filename="moe-film-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    print("--- Starting Training ---")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=[early_stop_cb, checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=wandb_logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        pl_module, 
        train_dataloaders=tr_loader, 
        val_dataloaders=va_loader,
    )

    wandb.finish()
