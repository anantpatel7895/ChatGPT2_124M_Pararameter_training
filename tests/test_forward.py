"""
Test forward pass equivalence
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2
from transformers import GPT2LMHeadModel

# Load models
print("Loading models...")
custom_model = GPT2.from_pretrained('gpt2')
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set to eval mode
custom_model.eval()
hf_model.eval()

# Create test input
torch.manual_seed(42)
input_ids = torch.randint(0, 50257, (1, 10))

print(f"\nTest input shape: {input_ids.shape}")
print(f"Input tokens: {input_ids[0, :5].tolist()}...")

# Forward pass
with torch.no_grad():
    custom_logits, _ = custom_model(input_ids)
    hf_output = hf_model(input_ids)
    hf_logits = hf_output.logits

print(f"\nCustom logits shape: {custom_logits.shape}")
print(f"HF logits shape: {hf_logits.shape}")

# Compare outputs
diff = torch.abs(custom_logits - hf_logits)
print(f"\nMax diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# Check predictions
custom_pred = torch.argmax(custom_logits[0, -1, :]).item()
hf_pred = torch.argmax(hf_logits[0, -1, :]).item()

print(f"\nCustom prediction: {custom_pred}")
print(f"HF prediction: {hf_pred}")
print(f"Predictions match: {custom_pred == hf_pred}")

# Check at different positions
print("\nPer-position max differences:")
for i in range(min(5, input_ids.shape[1])):
    pos_diff = torch.abs(custom_logits[0, i, :] - hf_logits[0, i, :]).max()
    print(f"  Position {i}: {pos_diff:.6f}")

if diff.max() < 1e-5:
    print("\n✅ Models produce identical outputs!")
elif diff.max() < 1e-3:
    print("\n✅ Models produce nearly identical outputs!")
else:
    print("\n⚠️ Models produce different outputs")
    
    # Debug: Check intermediate values
    print("\nDebugging intermediate values...")
    
    # Check embeddings
    with torch.no_grad():
        custom_tok_emb = custom_model.transformer.wte(input_ids)
        hf_tok_emb = hf_model.transformer.wte(input_ids)
        emb_diff = torch.abs(custom_tok_emb - hf_tok_emb).max()
        print(f"Token embedding diff: {emb_diff:.6f}")
        
        pos = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        custom_pos_emb = custom_model.transformer.wpe(pos)
        hf_pos_emb = hf_model.transformer.wpe(pos)
        pos_diff = torch.abs(custom_pos_emb - hf_pos_emb).max()
        print(f"Position embedding diff: {pos_diff:.6f}")
