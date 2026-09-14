"""
Check which weights actually use Conv1D in HuggingFace GPT-2
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import GPT2LMHeadModel

# Load HF model
model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

print("Checking module types in HuggingFace GPT-2:\n")
print("=" * 70)

# Check the first transformer block
block = model_hf.transformer.h[0]

print(f"attn.c_attn type: {type(block.attn.c_attn)}")
print(f"attn.c_attn weight shape: {block.attn.c_attn.weight.shape}")

print(f"\nattn.c_proj type: {type(block.attn.c_proj)}")
print(f"attn.c_proj weight shape: {block.attn.c_proj.weight.shape}")

print(f"\nmlp.c_fc type: {type(block.mlp.c_fc)}")
print(f"mlp.c_fc weight shape: {block.mlp.c_fc.weight.shape}")

print(f"\nmlp.c_proj type: {type(block.mlp.c_proj)}")
print(f"mlp.c_proj weight shape: {block.mlp.c_proj.weight.shape}")

print("\n" + "=" * 70)
print("\nConclusion:")
print("All c_attn, c_proj, c_fc use Conv1D")
print("Conv1D stores weights as (out_features, in_features)")
print("We need to transpose ALL of them for nn.Linear which uses (in_features, out_features)")
