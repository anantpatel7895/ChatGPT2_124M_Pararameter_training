"""
Check GELU implementation
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import GPT2LMHeadModel

# Load HF model
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Check the GELU activation
gelu = hf_model.transformer.h[0].mlp.act
print(f"HuggingFace GELU type: {type(gelu)}")
print(f"GELU module: {gelu}")

# Check if it's the same as nn.GELU()
test_input = torch.randn(2, 3, 768)

our_gelu = torch.nn.GELU()
hf_gelu_out = gelu(test_input)
our_gelu_out = our_gelu(test_input)

diff = torch.abs(hf_gelu_out - our_gelu_out).max()
print(f"\nDifference between HF GELU and nn.GELU(): {diff:.10f}")

if diff < 1e-6:
    print("✓ GELU implementations are identical")
else:
    print("✗ GELU implementations differ!")
    
    # Try the approximate version
    our_gelu_approx = torch.nn.GELU(approximate='tanh')
    our_gelu_approx_out = our_gelu_approx(test_input)
    diff_approx = torch.abs(hf_gelu_out - our_gelu_approx_out).max()
    print(f"Difference with GELU(approximate='tanh'): {diff_approx:.10f}")
