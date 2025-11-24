import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class Expert(nn.Module):
    """
    Simple feed-forward expert. Matches Notebook definition.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # Notebook uses GELU, script used SiLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TopKRouter(nn.Module):
    """
    Gating Network that selects the top-k experts for each input.
    """
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        # Select Top K
        top_k_vals, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        router_probs = F.softmax(top_k_vals, dim=1)
        return router_probs, top_k_indices, logits


class MoEClassifier(nn.Module):
    """
    Sparse MoE Head.
    """
    def __init__(self, input_dim, num_classes, num_experts=8, expert_hidden_dim=128, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TopKRouter(input_dim, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, num_classes) 
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (Batch, Embedding_Dim)
        """
        batch_size = x.size(0)
        router_probs, expert_indices, router_logits = self.router(x)
        
        # Output container
        # We look at the first expert to determine output size (num_classes)
        final_output = torch.zeros(batch_size, self.experts[0].net[-1].out_features, device=x.device)
        
        # Iterate through each k-th selected expert
        for k in range(self.top_k):
            # Indices of experts selected for the k-th position
            selected_experts = expert_indices[:, k]
            # Weights for the k-th position
            gate_weight = router_probs[:, k].unsqueeze(1)
            
            # For every expert, find which batch items selected it
            for expert_idx in range(self.num_experts):
                mask = (selected_experts == expert_idx)
                if mask.any():
                    # Process only the specific batch items for this expert
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    final_output[mask] += gate_weight[mask] * expert_output
                    
        return final_output, router_logits


class SBERT_MoE_Model(nn.Module):
    """
    Wrapper model.
    """
    def __init__(self, model_name_or_path='all-MiniLM-L6-v2', num_classes=10, num_experts=8, expert_hidden_dim=128, top_k=2, device=None):
        super().__init__()
        self.sbert = SentenceTransformer(model_name_or_path)
        
        # Calculate embedding dim
        try:
            sample = self.sbert.encode("hello world", convert_to_tensor=True)
            embedding_dim = sample.shape[-1]
        except Exception:
            embedding_dim = 384
            
        self.moe_head = MoEClassifier(
            input_dim=embedding_dim, 
            num_classes=num_classes, 
            num_experts=num_experts, 
            expert_hidden_dim=expert_hidden_dim, 
            top_k=top_k
        )
        
        if device is not None:
            self.to(device)

    def forward(self, text_input):
        # NOTE: In the notebook, features are detached to freeze backbone initially.
        # This behavior is largely controlled by the LightningModule optimizer logic,
        # but to match notebook exactly, we treat encode as 'frozen' features usually.
        features = self.sbert.encode(text_input, convert_to_tensor=True)
        
        # Ensure tensor on same device as head
        head_device = next(self.moe_head.parameters()).device
        features = features.to(head_device)
        
        logits, router_logits = self.moe_head(features)
        return logits, router_logits