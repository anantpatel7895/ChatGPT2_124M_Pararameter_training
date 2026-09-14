"""
Trace the weight loading process
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2
from transformers import GPT2LMHeadModel

# Load models
print("Loading models...")
custom_model = GPT2.from_pretrained('gpt2')
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

print("\n" + "=" * 70)
print("Checking attn.c_proj weights specifically")
print("=" * 70)

custom_weight = custom_model.transformer.h[0].attn.c_proj.weight
hf_weight = hf_model.transformer.h[0].attn.c_proj.weight

print(f"\nCustom c_proj weight shape: {custom_weight.shape}")
print(f"HF c_proj weight shape: {hf_weight.shape}")

print(f"\nCustom c_proj first 3x3:\n{custom_weight[:3, :3]}")
print(f"\nHF c_proj first 3x3:\n{hf_weight[:3, :3]}")
print(f"\nHF c_proj transposed first 3x3:\n{hf_weight.t()[:3, :3]}")

# Check if they match after transpose
diff_direct = torch.abs(custom_weight - hf_weight).max()
diff_transposed = torch.abs(custom_weight - hf_weight.t()).max()

print(f"\nDiff (direct): {diff_direct:.6f}")
print(f"Diff (transposed): {diff_transposed:.6f}")

if diff_transposed < 1e-6:
    print("\n✓ Custom model matches HF after transpose - This is CORRECT!")
    print("The from_pretrained method IS working properly.")
elif diff_direct < 1e-6:
    print("\n✗ Custom model matches HF without transpose - This is WRONG!")
    print("The transpose is not being applied correctly.")
else:
    print("\n✗ Custom model doesn't match HF - Something is broken!")
