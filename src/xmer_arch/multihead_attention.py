import torch
import torch.nn as nn
from torch.nn import functional as F


from ..config_models.gpt2_config import GPT2Config



class MultiheadAttention(nn.Module):
    """Multi-head self-attention matching HuggingFace GPT-2 naming convention"""

    def __init__(self, gpt2_config=GPT2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert gpt2_config.embd_dim % gpt2_config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        self.n_head = gpt2_config.n_head
        self.embd_dim = gpt2_config.embd_dim
        self.head_dim = gpt2_config.embd_dim // gpt2_config.n_head

        # fused QKV projection (like GPT-2 c_attn) - matching HF naming
        self.c_attn = nn.Linear(gpt2_config.embd_dim, 3 * gpt2_config.embd_dim)

        # output projection - matching HF naming
        self.c_proj = nn.Linear(gpt2_config.embd_dim, gpt2_config.embd_dim)
        
        # dropout
        self.attn_dropout = nn.Dropout(gpt2_config.dropout)
        self.resid_dropout = nn.Dropout(gpt2_config.dropout)
        
        # Causal mask buffer (not a parameter)
        self.register_buffer("bias", torch.tril(torch.ones(gpt2_config.max_seq_length, gpt2_config.max_seq_length))
                            .view(1, 1, gpt2_config.max_seq_length, gpt2_config.max_seq_length))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension

        # project to QKV and split into heads
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)  # (B, T, 3, n_head, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, n_head, T, head_dim)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, n_head, T, T)
        scaled_attn_scores = attn_scores / (self.head_dim ** 0.5)  # (B, n_head, T, T)

        # apply causal mask
        masked_scaled_attn_scores = scaled_attn_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # row-wise softmax and weighted sum of values
        attn_weights = F.softmax(masked_scaled_attn_scores, dim=-1)  # (B, n_head, T, T)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (B, n_head, T, head_dim)

        # combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, n_head, head_dim)
        attn_output = attn_output.view(B, T, C)  # (B, T, embd_dim)
        output = self.c_proj(attn_output)  # (B, T, embd_dim)
        output = self.resid_dropout(output)

        return output
