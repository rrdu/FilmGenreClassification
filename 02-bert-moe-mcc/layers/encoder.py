import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict
from .moe import MoEClassifier

class SimpleTokenizer:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, texts: Optional[List[str]] = None, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        if texts:
            self.build_vocab(texts)

    def build_vocab(self, texts: List[str]) -> None:
        from collections import Counter
        counter = Counter()
        for t in texts:
            tokens = self._tokenize(t)
            counter.update(tokens)
        self.token2id = {self.PAD: 0, self.UNK: 1}
        idx = 2
        for token, freq in counter.most_common():
            if freq >= self.min_freq and token not in self.token2id:
                self.token2id[token] = idx
                idx += 1
        self.id2token = {i: t for t, i in self.token2id.items()}

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().strip().split()

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None
                    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        # Ensure PAD and UNK exist
        if self.PAD not in self.token2id or self.UNK not in self.token2id:
            self.token2id = {self.PAD: 0, self.UNK: 1}
            self.id2token = {0: self.PAD, 1: self.UNK}

        token_ids = []
        for t in texts:
            ids = []
            for tok in self._tokenize(t):
                ids.append(self.token2id.get(tok, self.token2id[self.UNK]))
            token_ids.append(ids)

        if max_length is None:
            max_length = max((len(x) for x in token_ids), default=1)
        batch_size = len(token_ids)
        ids_tensor = torch.full((batch_size, max_length), fill_value=self.token2id[self.PAD], dtype=torch.long)
        attn_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        for i, seq in enumerate(token_ids):
            L = min(len(seq), max_length)
            if L > 0:
                ids_tensor[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
                attn_mask[i, :L] = True
        return ids_tensor, attn_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]

class CustomTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_dim = proj_dim if proj_dim is not None else emb_dim
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, self.proj_dim),
            nn.Tanh()
        )

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        device = input_ids.device
        x = self.token_emb(input_ids) * math.sqrt(self.emb_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)

        key_padding_mask = ~attention_mask
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        attn = attention_mask.float().unsqueeze(-1)
        summed = (x * attn).sum(dim=1)
        lengths = attn.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / lengths

        embeddings = self.pool_proj(mean_pooled)
        return embeddings

class CustomSBERTLikeEncoder(nn.Module):
    def __init__(
        self,
        texts_for_vocab: Optional[List[str]] = None,
        vocab_size: Optional[int] = None,
        emb_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        max_seq_len: int = 128,
        proj_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = SimpleTokenizer(texts_for_vocab) if texts_for_vocab is not None else SimpleTokenizer()
        if vocab_size is not None and len(self.tokenizer.token2id) < 2:
            for i in range(2, vocab_size):
                self.tokenizer.token2id[f"[T{i}]"] = i
            self.tokenizer.id2token = {i: t for t, i in self.tokenizer.token2id.items()}

        vocab_size_final = max(len(self.tokenizer.token2id), 2)
        self.encoder = CustomTransformerEncoder(
            vocab_size=vocab_size_final,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            proj_dim=proj_dim,
            dropout=dropout
        )

    def get_sentence_embedding_dimension(self) -> int:
        return self.encoder.proj_dim

    def encode(self, text_input: List[str], convert_to_tensor: bool = True, device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(text_input, str):
            text_input = [text_input]
        if len(self.tokenizer.token2id) <= 2:
            self.tokenizer.build_vocab(text_input)

        input_ids, attn_mask = self.tokenizer.encode_batch(text_input)
        if device is None:
            device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            embeddings = self.encoder(input_ids, attn_mask)
        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy()

class SBERT_MoE_Model(nn.Module):
    def __init__(self,
                 model_name: str = 'custom-small',
                 num_classes: int = 5,
                 num_experts: int = 8,
                 expert_hidden_dim: int = 128,
                 top_k: int = 2,
                 vocab_texts: Optional[List[str]] = None,
                 encoder_kwargs: Optional[dict] = None):
        super().__init__()
        if encoder_kwargs is None:
            encoder_kwargs = {}
        self.sbert = CustomSBERTLikeEncoder(texts_for_vocab=vocab_texts, **encoder_kwargs)

        target_device = next(self.sbert.encoder.parameters()).device

        embedding_dim = self.sbert.get_sentence_embedding_dimension()
        self.moe_head = MoEClassifier(
            input_dim=embedding_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k
        )

        self.moe_head.to(target_device)

    def forward(self, text_input: List[str]):
        # Tokenize & encode then pass through MoE head
        input_ids, attn_mask = self.sbert.tokenizer.encode_batch(text_input)
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        embeddings = self.sbert.encoder(input_ids, attn_mask)   # -> (B, embedding_dim)
        logits, router_logits = self.moe_head(embeddings)
        return logits, router_logits