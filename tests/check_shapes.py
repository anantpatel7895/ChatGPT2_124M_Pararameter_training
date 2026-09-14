"""
Check if our Linear layer shapes match what we expect after transpose
"""
import torch
import torch.nn as nn
import sys; sys.path.append("..")
from src.config_models.gpt2_config import GPT2Config
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2

# Create config
config = GPT2Config(
    vocab_size=50257,
    max_seq_length=1024,
    embd_dim=768,
    n_head=12,
    n_xmer_layer=12,
    dropout=0.1
)

# Create model
model = GPT2(config)

# Get first transformer block
block = model.transformer.h[0]

print("Our model Linear layer shapes:\n")
print("=" * 70)
print(f"attn.c_attn weight shape: {block.attn.c_attn.weight.shape}")
print(f"  Expected: [2304, 768] (out, in) for Linear")
print(f"  HF Conv1D: [768, 2304] - needs transpose")

print(f"\nattn.c_proj weight shape: {block.attn.c_proj.weight.shape}")
print(f"  Expected: [768, 768] (out, in) for Linear")
print(f"  HF Conv1D: [768, 768] - needs transpose (but same shape!)")

print(f"\nmlp.c_fc weight shape: {block.mlp.c_fc.weight.shape}")
print(f"  Expected: [3072, 768] (out, in) for Linear")
print(f"  HF Conv1D: [768, 3072] - needs transpose")

print(f"\nmlp.c_proj weight shape: {block.mlp.c_proj.weight.shape}")
print(f"  Expected: [768, 3072] (out, in) for Linear")
print(f"  HF Conv1D: [3072, 768] - needs transpose")

print("\n" + "=" * 70)
print("\nConclusion: ALL Conv1D weights need transpose, even attn.c_proj!")
