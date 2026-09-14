import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForwardMLP(nn.Module):
    """Two-layer MLP matching HuggingFace GPT-2 naming convention"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForwardMLP, self).__init__()
        # Matching HF naming: c_fc (fully connected) and c_proj (projection)
        self.c_fc = nn.Linear(input_dim, hidden_dim)
        # Use tanh approximation like HuggingFace GPT-2 (NewGELUActivation)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x