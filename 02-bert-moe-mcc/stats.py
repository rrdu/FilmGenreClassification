import torch
from layers.encoder import SBERT_MoE_Model

# Example — use the same parameters as your train.py
num_classes = 8
num_experts = 8
expert_hidden_dim = 128
top_k = 2
encoder_kwargs = dict(
    emb_dim=256,
    n_layers=3,
    n_heads=4,
    ff_dim=512,
    max_seq_len=256,
)

# Instantiate your model
model = SBERT_MoE_Model(
    num_classes=num_classes,
    num_experts=num_experts,
    expert_hidden_dim=expert_hidden_dim,
    top_k=top_k,
    vocab_texts=None,  # can pass dummy or real vocab_texts
    encoder_kwargs=encoder_kwargs,
)

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
