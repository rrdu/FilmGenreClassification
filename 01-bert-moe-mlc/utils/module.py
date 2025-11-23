import pandas as pd
import pathlib as Path
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import Dataset
from torchmetrics.classification import MultilabelF1Score


class MoE_LightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, num_experts, learning_rate=1e-3, aux_loss_weight=0.1, pos_weight=None, threshold=0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'pos_weight']) # Don't save huge tensors to hparams
        
        self.backbone = model.sbert
        self.head = model.moe_head
        self.learning_rate = learning_rate
        self.num_experts = num_experts
        self.aux_loss_weight = aux_loss_weight
        
        if pos_weight is not None:
            # Ensure it's a tensor
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight)
            
            # Register as buffer so it moves to device automatically with the model
            self.register_buffer('pos_weight', pos_weight)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, threshold=threshold, average='micro')

    def forward(self, x):
        # 1. Tokenize the raw text
        # This converts ["Hello world"] into {'input_ids': ..., 'attention_mask': ...}
        features = self.backbone.tokenize(x)
        
        # 2. Move inputs to the correct device (GPU/MPS)
        # LightningModule provides self.device
        features = {key: value.to(self.device) for key, value in features.items()}
        
        # 3. Pass through SBERT backbone
        # calling .forward() or __call__() allows gradients to flow (unlike .encode())
        out = self.backbone(features)
        
        # 4. Extract the sentence embedding
        embeddings = out['sentence_embedding']
        
        # 5. Pass through MoE Head
        return self.head(embeddings)

    def _compute_load_balancing_loss(self, router_logits):
        probs = F.softmax(router_logits, dim=1)
        mean_probs = probs.mean(dim=0)
        aux_loss = (mean_probs ** 2).sum() * self.num_experts
        return aux_loss

    def training_step(self, batch, batch_idx):
        texts, targets = batch
        logits, router_logits = self(texts)
        
        cls_loss = self.criterion(logits, targets)
        aux_loss = self._compute_load_balancing_loss(router_logits)
        
        total_loss = cls_loss + (self.aux_loss_weight * aux_loss)
        
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        texts, targets = batch
        
        # Forward pass (we don't need router logits for validation metrics)
        logits, _ = self(texts)
        
        # Calculate Loss
        val_loss = self.criterion(logits, targets)
        
        # Update F1 Score
        self.val_f1(logits, targets)
        
        # Log metrics so the Scheduler can find 'val_loss'
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def configure_optimizers(self):
        # 1. Initialize Optimizer with ONLY the Head (Backbone added later by Callback)
        optimizer = torch.optim.AdamW(
            self.head.parameters(), 
            lr=1e-3, 
            weight_decay=0.01
        )
        
        # 2. Use ReduceLROnPlateau (Safe for dynamic unfreezing)
        # It waits for 'val_loss' to stop improving, then lowers LR
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
                "monitor": "val_loss", # Required for ReduceLROnPlateau
                "interval": "epoch",
                "frequency": 1
            }
        }
        

class IMDBDataset(Dataset):
    def __init__(self, data_dir_path, filename, class_names, text_col='description', label_col='genre_list'):
        """
        Args:
            data_dir_path (Path or str): Relative path to data directory.
            filename (str): The CSV filename (e.g., 'train.csv').
            class_names (list): List of valid classes (ORDER MATTERS).
            text_col (str): Column name for text.
            label_col (str): Column name for labels.
        """
        # 1. Setup Path safely
        self.data_path = Path(data_dir_path) / filename
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found at: {self.data_path.resolve()}")
            
        print(f"Loading data from {self.data_path.name}...")
        self.df = pd.read_csv(self.data_path)
        
        # 2. Process Text (Handle NaNs, ensure strings)
        self.texts = self.df[text_col].fillna("").astype(str).tolist()
        
        # 3. Process Labels (Multi-hot Encoding)
        # Create map: "Action" -> 0, "Drama" -> 1, etc.
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.num_classes = len(class_names)
        self.labels = []
        
        # Track unseen genres for safety warning
        unseen_genres = set()
        
        for genre_str in self.df[label_col]:
            # Initialize zero vector [0.0, 0.0, ...]
            label_vec = torch.zeros(self.num_classes, dtype=torch.float)
            
            if pd.notna(genre_str):
                # Split "Action, Drama" -> ["Action", "Drama"]
                # .strip() removes whitespace around words
                current_genres = [g.strip() for g in str(genre_str).split(',')]
                
                for genre in current_genres:
                    if genre in self.class_to_idx:
                        idx = self.class_to_idx[genre]
                        label_vec[idx] = 1.0
                    else:
                        # Track genres that don't match our class list
                        unseen_genres.add(genre)
            
            self.labels.append(label_vec)

        # 4. Warning System
        # If this is the test set, and it has weird genres not in train, warn the user.
        if unseen_genres:
            print(f"⚠️  WARNING in {filename}: Found {len(unseen_genres)} genres not in the provided class list.")
            print(f"   Examples of ignored genres: {list(unseen_genres)[:5]}")
            print(f"   (These were ignored during label creation)")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    @staticmethod
    def discover_classes(data_dir_path, filename, label_col='genre_list'):
        """
        Static utility to scan a CSV and return sorted unique class names.
        Use this ONCE on your TRAINING set only.
        """
        path = Path(data_dir_path) / filename
        if not path.exists():
             raise FileNotFoundError(f"File not found at: {path.resolve()}")

        df = pd.read_csv(path)
        
        # Split by comma, explode list to rows, strip whitespace, find unique
        genres = df[label_col].dropna().astype(str).str.split(',').explode().str.strip().unique()
        
        return sorted(list(genres))