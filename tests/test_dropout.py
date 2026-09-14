"""
Check if dropout is properly disabled in eval mode
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2

# Load model
model = GPT2.from_pretrained('gpt2')

print("Checking dropout modules:")
print("=" * 70)

# Check if model is in training mode
print(f"\nModel training mode: {model.training}")

# Set to eval
model.eval()
print(f"After model.eval(): {model.training}")

# Check dropout modules
dropout_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        dropout_modules.append((name, module.training, module.p))

print(f"\nFound {len(dropout_modules)} Dropout modules:")
for name, training, p in dropout_modules[:10]:
    status = "TRAINING" if training else "EVAL"
    print(f"  {name:60s} p={p:.2f} [{status}]")

print("\nTest: Run same input twice and check if output is identical")
model.eval()
torch.manual_seed(42)
input_ids = torch.randint(0, 50257, (1, 5))

with torch.no_grad():
    logits1, _ = model(input_ids)
    logits2, _ = model(input_ids)

diff = torch.abs(logits1 - logits2).max()
print(f"\nDifference between two runs: {diff:.10f}")

if diff == 0:
    print("✅ Output is deterministic - dropout is properly disabled!")
else:
    print("⚠️ Output is non-deterministic - dropout might still be active!")
