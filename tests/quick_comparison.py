"""
Quick example showing that our GPT-2 matches HuggingFace
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 70)
print("Comparing Custom GPT-2 vs HuggingFace GPT-2")
print("=" * 70)

# Load models
print("\nLoading models...")
custom_model = GPT2.from_pretrained('gpt2')
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

custom_model.eval()
hf_model.eval()

# Test prompt
prompt = "The meaning of life is"
print(f"\nPrompt: '{prompt}'")

# Encode
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Get predictions from both models
with torch.no_grad():
    custom_logits, _ = custom_model(input_ids)
    hf_logits = hf_model(input_ids).logits

# Get top 5 predictions from each model
custom_top5 = torch.topk(custom_logits[0, -1, :], 5)
hf_top5 = torch.topk(hf_logits[0, -1, :], 5)

print("\n" + "=" * 70)
print("Top 5 Next Token Predictions")
print("=" * 70)

print("\n🔧 Custom GPT-2:")
for i, (score, token_id) in enumerate(zip(custom_top5.values, custom_top5.indices), 1):
    token = tokenizer.decode([token_id])
    print(f"  {i}. '{token}' (score: {score:.4f})")

print("\n🤗 HuggingFace GPT-2:")
for i, (score, token_id) in enumerate(zip(hf_top5.values, hf_top5.indices), 1):
    token = tokenizer.decode([token_id])
    print(f"  {i}. '{token}' (score: {score:.4f})")

# Compare outputs
diff = torch.abs(custom_logits - hf_logits)
print("\n" + "=" * 70)
print("Comparison Statistics")
print("=" * 70)
print(f"\nMax absolute difference: {diff.max():.8f}")
print(f"Mean absolute difference: {diff.mean():.8f}")
print(f"Total parameters: {sum(p.numel() for p in custom_model.parameters()):,}")

# Check if predictions match
custom_pred = torch.argmax(custom_logits[0, -1, :]).item()
hf_pred = torch.argmax(hf_logits[0, -1, :]).item()

if custom_pred == hf_pred:
    print(f"\n✅ Top predictions match: '{tokenizer.decode([custom_pred])}'")
else:
    print(f"\n⚠️ Top predictions differ!")

if diff.max() < 1e-4:
    print("✅ Models produce virtually identical outputs!")
    print("\n🎉 Implementation is correct and matches HuggingFace GPT-2!")
else:
    print(f"\n⚠️ Outputs differ by {diff.max():.6f}")

print("\n" + "=" * 70)
