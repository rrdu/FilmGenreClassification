import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from transformers import get_linear_schedule_with_warmup
from typing import List
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class MoE_LightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, num_experts, learning_rate=1e-3, aux_loss_weight=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Keep references to encoder (CustomSBERTLikeEncoder) and head (MoEClassifier)
        # 'model' is expected to be SBERT_MoE_Model instance
        self.backbone = model.sbert
        self.head = model.moe_head
        self.learning_rate = learning_rate
        
        self.num_experts = num_experts
        self.aux_loss_weight = aux_loss_weight
        
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')

    def forward(self, texts: List[str]):
        # Use tokenizer -> encoder -> head
        # texts: List[str]
        input_ids, attn_mask = self.backbone.tokenizer.encode_batch(texts)
        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)
        embeddings = self.backbone.encoder(input_ids, attn_mask)  # (B, emb_dim)
        logits, router_logits = self.head(embeddings)
        return logits, router_logits

    def _compute_load_balancing_loss(self, router_logits):
        """
        Encourages the router to send equal traffic to all experts.
        """
        probs = F.softmax(router_logits, dim=1) # [B, num_experts]
        mean_probs = probs.mean(dim=0) # [num_experts]
        aux_loss = (mean_probs ** 2).sum() * self.num_experts
        return aux_loss

    def training_step(self, batch, batch_idx):
        texts, targets = batch
        # texts is a list of strings (DataLoader default collate)
        logits, router_logits = self(texts)
        
        cls_loss = self.criterion(logits, targets.to(self.device))
        aux_loss = self._compute_load_balancing_loss(router_logits)
        total_loss = cls_loss + (self.aux_loss_weight * aux_loss)
        
        self.log("train_loss", total_loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_aux_loss", aux_loss, prog_bar=False, on_step=True, on_epoch=False)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        texts, targets = batch
        logits, router_logits = self(texts)
        loss = self.criterion(logits, targets.to(self.device))
        acc_val = self.val_acc(logits, targets.to(self.device))

        # expert usage
        probs = F.softmax(router_logits, dim=1)
        mean_probs = probs.mean(dim=0)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc_val, prog_bar=True, on_step=False, on_epoch=True)
        for i,p in enumerate(mean_probs):
            self.log(f"expert/mean_prob_{i}", p, on_epoch=True)
        
        # log the scalar returned
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc_val, prog_bar=True, on_step=False, on_epoch=True)

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        """
        AdamW optimizer + Hugging Face Transformers linear warmup/decay scheduler.
        Requires `transformers` to be installed.
        """
        # 1️⃣ Parameter groups with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        wd_params, no_wd_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": wd_params, "weight_decay": 0.01},
                {"params": no_wd_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

        # 2️⃣ Compute warmup/total steps using Lightning’s trainer
        total_steps = getattr(self.trainer, "estimated_stepping_batches", 10000)
        warmup_frac = float(self.hparams.get("warmup_frac", 0.1)) if hasattr(self, "hparams") else 0.1
        num_warmup_steps = int(total_steps * warmup_frac)

        # 3️⃣ Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

        # 4️⃣ Return optimizer + scheduler dict
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class IMDBDataset(Dataset):
    def __init__(self, data_dir_path, filename, class_names, text_col='description', label_col='csv_genre'):
        self.data_path = Path(data_dir_path) / filename
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found at: {self.data_path.resolve()}")
            
        print(f"Loading data from {self.data_path.name}...")
        self.df = pd.read_csv(self.data_path)
        
        self.texts = self.df[text_col].fillna("").astype(str).tolist()
        
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.labels = []
        
        for genre_raw in self.df[label_col]:
            genre_str = str(genre_raw).strip()
            if genre_str in self.class_to_idx:
                self.labels.append(self.class_to_idx[genre_str])
            else:
                raise ValueError(f"Unknown genre label found: '{genre_str}' not in class list.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return self.texts[idx], label_tensor

    @staticmethod
    def discover_classes(data_dir_path, filename, label_col='csv_genre'):
        path = Path(data_dir_path) / filename
        df = pd.read_csv(path)
        genres = df[label_col].dropna().astype(str).str.strip().unique()
        return sorted(list(genres))