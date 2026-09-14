import torch
import torch.nn as nn   

from ..config_models.gpt2_config import GPT2Config
from .multihead_attention import MultiheadAttention
from .feed_forward_mlp import FeedForwardMLP


class XMERLayer(nn.Module):
    """Transformer block with residual connections matching GPT-2 architecture"""
    
    def __init__(self, gpt2_config=GPT2Config, *args, **kwargs):    
        super().__init__(*args, **kwargs)

        # layer normalization before attention (pre-norm architecture)
        self.ln_1 = nn.LayerNorm(gpt2_config.embd_dim)

        # multi-head self-attention layer
        self.attn = MultiheadAttention(gpt2_config)

        # layer normalization before feedforward layer
        self.ln_2 = nn.LayerNorm(gpt2_config.embd_dim) 

        # two-layer MLP with GELU activation
        self.mlp = FeedForwardMLP(gpt2_config.embd_dim, gpt2_config.embd_dim * 4, gpt2_config.dropout)

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x
