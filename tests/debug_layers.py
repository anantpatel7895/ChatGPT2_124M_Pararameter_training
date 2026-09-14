"""
Layer-by-layer comparison to find where differences accumulate
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2
from transformers import GPT2LMHeadModel

# Load models
print("Loading models...")
custom_model = GPT2.from_pretrained('gpt2')
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

custom_model.eval()
hf_model.eval()

# Test input
torch.manual_seed(42)
input_ids = torch.randint(0, 50257, (1, 5))
print(f"Input: {input_ids}")

print("\n" + "=" * 70)
print("Layer-by-layer comparison")
print("=" * 70)

with torch.no_grad():
    # Embeddings
    pos = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    
    custom_tok_emb = custom_model.transformer.wte(input_ids)
    hf_tok_emb = hf_model.transformer.wte(input_ids)
    print(f"\nToken embeddings diff: {torch.abs(custom_tok_emb - hf_tok_emb).max():.6f}")
    
    custom_pos_emb = custom_model.transformer.wpe(pos)
    hf_pos_emb = hf_model.transformer.wpe(pos)
    print(f"Position embeddings diff: {torch.abs(custom_pos_emb - hf_pos_emb).max():.6f}")
    
    custom_x = custom_model.transformer.drop(custom_tok_emb + custom_pos_emb)
    hf_x = hf_model.transformer.drop(hf_tok_emb + hf_pos_emb)
    print(f"After embedding + dropout diff: {torch.abs(custom_x - hf_x).max():.6f}")
    
    # Pass through each transformer layer
    for i in range(min(3, len(custom_model.transformer.h))):
        custom_x = custom_model.transformer.h[i](custom_x)
        hf_x = hf_model.transformer.h[i](hf_x)
        diff = torch.abs(custom_x - hf_x).max()
        print(f"After transformer block {i}: {diff:.6f}")
        
        if diff > 0.001:
            # Debug this layer
            print(f"  ⚠️ Large difference detected! Investigating...")
            
            # Re-run with intermediate checks
            custom_x_debug = custom_model.transformer.drop(custom_tok_emb + custom_pos_emb)
            hf_x_debug = hf_model.transformer.drop(hf_tok_emb + hf_pos_emb)
            
            for j in range(i):
                custom_x_debug = custom_model.transformer.h[j](custom_x_debug)
                hf_x_debug = hf_model.transformer.h[j](hf_x_debug)
            
            # Check layer norm 1
            custom_ln1 = custom_model.transformer.h[i].ln_1(custom_x_debug)
            hf_ln1 = hf_model.transformer.h[i].ln_1(hf_x_debug)
            print(f"    After LN1: {torch.abs(custom_ln1 - hf_ln1).max():.6f}")
            
            # Check attention
            custom_attn_out = custom_model.transformer.h[i].attn(custom_ln1)
            hf_attn_out = hf_model.transformer.h[i].attn(hf_ln1)[0]  # HF returns tuple
            print(f"    After attention: {torch.abs(custom_attn_out - hf_attn_out).max():.6f}")
            
            break
