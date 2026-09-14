"""
Debug script to check if weights are being loaded correctly
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2
from transformers import GPT2LMHeadModel


def compare_weights():
    """Compare weights between custom and HF models"""
    print("=" * 70)
    print("Debugging Weight Loading")
    print("=" * 70)
    
    # Load both models
    print("\nLoading custom GPT2...")
    custom_model = GPT2.from_pretrained('gpt2')
    
    print("\nLoading HuggingFace GPT2...")
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Get state dicts
    custom_sd = custom_model.state_dict()
    hf_sd = hf_model.state_dict()
    
    print("\n" + "=" * 70)
    print("Comparing weights")
    print("=" * 70)
    
    # Compare a few key weights
    test_keys = [
        ('transformer.wte.weight', 'transformer.wte.weight'),
        ('transformer.wpe.weight', 'transformer.wpe.weight'),
        ('transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.weight'),
        ('transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_attn.weight'),
        ('transformer.h.0.attn.c_proj.weight', 'transformer.h.0.attn.c_proj.weight'),
        ('transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_fc.weight'),
        ('transformer.ln_f.weight', 'transformer.ln_f.weight'),
    ]
    
    print("\nChecking key weight matrices:\n")
    all_match = True
    
    for custom_key, hf_key in test_keys:
        if custom_key in custom_sd and hf_key in hf_sd:
            custom_weight = custom_sd[custom_key]
            hf_weight = hf_sd[hf_key]
            
            # Print shapes
            print(f"\n{custom_key}:")
            print(f"  Custom shape: {custom_weight.shape}")
            print(f"  HF shape:     {hf_weight.shape}")
            
            # Check if shapes match
            if custom_weight.shape == hf_weight.shape:
                diff = torch.abs(custom_weight - hf_weight).max().item()
                match = diff < 1e-6
                all_match = all_match and match
                
                status = "✓" if match else "✗"
                print(f"  {status} Diff: {diff:.2e}")
            elif custom_weight.shape == hf_weight.shape[::-1]:
                # Needs transpose
                diff = torch.abs(custom_weight - hf_weight.t()).max().item()
                match = diff < 1e-6
                all_match = all_match and match
                
                status = "✓" if match else "✗"
                print(f"  {status} Diff (after transpose): {diff:.2e}")
            else:
                print(f"  ✗ Shape mismatch!")
                all_match = False
        else:
            print(f"✗ {custom_key:50s} MISSING")
            all_match = False
    
    print("\n" + "=" * 70)
    if all_match:
        print("✅ All weights match! The implementation is correct!")
    else:
        print("❌ Some weights don't match. Checking details...")
        
        # Check shapes
        print("\nParameter shapes comparison:")
        print(f"\nCustom model parameters: {len(custom_sd)}")
        print(f"HuggingFace parameters: {len(hf_sd)}")
        
        # Find mismatched keys
        custom_keys = set(custom_sd.keys())
        hf_keys = set(hf_sd.keys())
        
        # Filter out bias keys
        custom_keys = {k for k in custom_keys if not k.endswith('.attn.bias')}
        hf_keys = {k for k in hf_keys if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')}
        
        only_custom = custom_keys - hf_keys
        only_hf = hf_keys - custom_keys
        
        if only_custom:
            print(f"\nKeys only in custom model ({len(only_custom)}):")
            for k in sorted(only_custom)[:10]:
                print(f"  - {k}")
        
        if only_hf:
            print(f"\nKeys only in HF model ({len(only_hf)}):")
            for k in sorted(only_hf)[:10]:
                print(f"  - {k}")
    
    print("=" * 70)


if __name__ == "__main__":
    compare_weights()
