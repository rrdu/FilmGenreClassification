import os
import pathlib as Path
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    torch.set_float32_matmul_precision('medium')
except Exception:
    pass

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from torchmetrics.classification import MultilabelF1Score
from sentence_transformers import SentenceTransformer

from layers.moe import SBERT_MoE_Model
from utils.module import IMDBDataset, MoE_LightningModule

k = 2
num_experts = 10
expert_hidden_dim = 128


if __name__ == '__main__':
    DATA_DIR = Path('../data/imdb_arh_trimmed')
    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, 'imdb_arh_train.csv')
    NUM_WORKERS = 4

    tr_ds = IMDBDataset(data_dir_path=DATA_DIR, filename='imdb_arh_train.csv', class_names=CLASS_NAMES)
    va_ds = IMDBDataset(data_dir_path=DATA_DIR, filename='imdb_arh_val.csv', class_names=CLASS_NAMES)
    te_ds = IMDBDataset(data_dir_path=DATA_DIR, filename='imdb_arh_test.csv', class_names=CLASS_NAMES)

    tr_loader = DataLoader(tr_ds, batch_size=32, num_workers=NUM_WORKERS, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, num_workers=NUM_WORKERS, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=64, num_workers=NUM_WORKERS, shuffle=False)
