import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    A simple Feed-Forward Network acting as a single 'Expert'.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
        top_k_vals, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        router_probs = F.softmax(top_k_vals, dim=1)
        return router_probs, top_k_indices, logits

class MoEClassifier(nn.Module):
    """
    The Mixture of Experts Classification Head.
    """
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2, expert_hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TopKRouter(input_dim, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, num_classes) 
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        router_probs, expert_indices, router_logits = self.router(x)
        
        final_output = torch.zeros(batch_size, self.experts[0].net[-1].out_features, device=x.device)
        
        for k in range(self.top_k):
            selected_experts = expert_indices[:, k]
            gate_weight = router_probs[:, k].unsqueeze(1)  # (B,1)
            
            for expert_idx in range(self.num_experts):
                mask = (selected_experts == expert_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    # gate_weight[mask] has shape (n_selected, 1)
                    final_output[mask] += gate_weight[mask] * expert_output
                    
        return final_output, router_logits