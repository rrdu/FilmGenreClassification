import pandas as pd
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import Dataset
from torchmetrics.classification import MultilabelF1Score
import numpy as np

import pandas as pd
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import Dataset
from torchmetrics.classification import MultilabelF1Score
import numpy as np

class MoE_LightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes: int,
        num_experts: int = 4,
        learning_rate: float = 1e-3,
        aux_loss_weight: float = 0.1,
        pos_weight=None,
        threshold: float = 0.5,
        weight_decay: float = 0.01,
        finetune_backbone: bool = False,
        multilabel: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.backbone = getattr(model, "sbert", None)
        self.head = getattr(model, "moe_head", None)

        if self.backbone is None or self.head is None:
            raise ValueError("Provided `model` must have `.sbert` and `.moe_head` attributes")

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.aux_loss_weight = aux_loss_weight
        self.threshold = threshold
        self.weight_decay = weight_decay
        self.finetune_backbone = finetune_backbone
        self.num_experts = num_experts

        # --- Loss Function ---
        if pos_weight is not None:
             self.register_buffer("pos_weight", torch.tensor(pos_weight))
             self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
             self.criterion = nn.BCEWithLogitsLoss()

        # --- Metrics ---
        self.val_f1 = MultilabelF1Score(num_labels=self.num_classes, average='micro', threshold=self.threshold)
        self.train_f1 = MultilabelF1Score(num_labels=self.num_classes, average='micro', threshold=self.threshold)

    def forward(self, texts):
        # Manually run forward pass since self.model is not registered
        # 1. Encode via Backbone
        features = self.backbone.encode(texts, convert_to_tensor=True)
        # Ensure features are on the correct device
        features = features.to(self.device)
        
        # 2. Pass through MoE Head
        logits, router_logits = self.head(features)
        return logits, router_logits

    def _compute_load_balancing_loss(self, router_logits):
        probs = F.softmax(router_logits, dim=1)
        mean_probs = probs.mean(dim=0)
        aux_loss = (mean_probs ** 2).sum() * self.num_experts
        return aux_loss

    def training_step(self, batch, batch_idx):
        texts, targets = batch
        targets = targets.to(self.device).float()
        
        logits, router_logits = self.forward(texts)
        
        cls_loss = self.criterion(logits, targets)
        aux_loss = self._compute_load_balancing_loss(router_logits)
        loss = cls_loss + self.aux_loss_weight * aux_loss

        probs = torch.sigmoid(logits)
        self.train_f1.update(probs, targets.int())
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        texts, targets = batch
        targets = targets.to(self.device).float()
        
        logits, router_logits = self.forward(texts)
        val_loss = self.criterion(logits, targets)
        
        probs = torch.sigmoid(logits)
        self.val_f1.update(probs, targets.int())

        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        self.log("val_f1", f1, prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.head.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


class IMDBDataset(Dataset):
    def __init__(self, data_dir_path, filename, class_names=None, text_col='description', label_col='genre_list', multilabel=True):
        super().__init__()
        data_dir = pathlib.Path(data_dir_path)
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path.resolve()}")
        self.df = pd.read_csv(path)
        self.text_col = text_col
        self.label_col = label_col
        self.multilabel = multilabel

        if class_names is None:
            self.class_names = self.discover_classes(data_dir, filename, label_col=self.label_col)
        else:
            self.class_names = class_names

        self.texts = self.df[self.text_col].fillna("").astype(str).tolist()
        self.labels = []
        for i, row in self.df.iterrows():
            lbl = row.get(self.label_col, "")
            if pd.isna(lbl):
                lbl = ""
            if self.multilabel:
                parts = [p.strip() for p in str(lbl).split(",") if p.strip()]
                vec = np.zeros(len(self.class_names), dtype=np.float32)
                for p in parts:
                    if p in self.class_names:
                        vec[self.class_names.index(p)] = 1.0
                self.labels.append(vec)
            else:
                idx = -1
                s = str(lbl).strip()
                if s in self.class_names:
                    idx = self.class_names.index(s)
                self.labels.append(np.int64(idx))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.multilabel:
            return text, torch.from_numpy(label).float()
        else:
            return text, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def discover_classes(data_dir_path, filename, label_col='genre_list'):
        path = pathlib.Path(data_dir_path) / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path.resolve()}")
        df = pd.read_csv(path)
        genres = df[label_col].dropna().astype(str).str.split(',').explode().str.strip().unique()
        return sorted(list(genres))