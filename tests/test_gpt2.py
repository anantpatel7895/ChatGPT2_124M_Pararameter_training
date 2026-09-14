"""
Test script to verify GPT2 implementation matches HuggingFace
"""
import torch
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_models.gpt2_config import GPT2Config
from src.xmer_arch.gpt2 import GPT2


def test_model_creation():
    """Test creating a GPT2 model from scratch"""
    print("=" * 50)
    print("Test 1: Creating GPT2 model from scratch")
    print("=" * 50)
    
    config = GPT2Config(
        n_xmer_layer=12,
        n_head=12,
        embd_dim=768,
        vocab_size=50257,
        max_seq_length=1024,
        dropout=0.1
    )
    
    model = GPT2(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected for GPT2-124M: ~124,000,000")
    
    # Test forward pass
    batch_size = 2
    seq_length = 10
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    with torch.no_grad():
        logits, loss = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: (batch={batch_size}, seq={seq_length}, vocab={config.vocab_size})")
    
    assert logits.shape == (batch_size, seq_length, config.vocab_size), "Output shape mismatch!"
    print("✓ Model creation and forward pass successful!\n")


def test_model_architecture():
    """Test that the model architecture matches expected GPT2 structure"""
    print("=" * 50)
    print("Test 2: Checking model architecture")
    print("=" * 50)
    
    config = GPT2Config(n_xmer_layer=12, n_head=12, embd_dim=768)
    model = GPT2(config)
    
    # Check key components
    assert hasattr(model, 'transformer'), "Missing transformer attribute"
    assert hasattr(model.transformer, 'wte'), "Missing token embeddings"
    assert hasattr(model.transformer, 'wpe'), "Missing position embeddings"
    assert hasattr(model.transformer, 'h'), "Missing transformer blocks"
    assert hasattr(model.transformer, 'ln_f'), "Missing final layer norm"
    assert hasattr(model, 'lm_head'), "Missing language model head"
    
    print(f"Number of transformer layers: {len(model.transformer.h)}")
    print(f"Embedding dimension: {config.embd_dim}")
    print(f"Number of attention heads: {config.n_head}")
    
    # Check first transformer block structure
    first_block = model.transformer.h[0]
    assert hasattr(first_block, 'ln_1'), "Missing first layer norm"
    assert hasattr(first_block, 'attn'), "Missing attention layer"
    assert hasattr(first_block, 'ln_2'), "Missing second layer norm"
    assert hasattr(first_block, 'mlp'), "Missing MLP layer"
    
    # Check attention layer naming
    attn = first_block.attn
    assert hasattr(attn, 'c_attn'), "Missing c_attn (should match HF naming)"
    assert hasattr(attn, 'c_proj'), "Missing c_proj (should match HF naming)"
    
    # Check MLP layer naming
    mlp = first_block.mlp
    assert hasattr(mlp, 'c_fc'), "Missing c_fc (should match HF naming)"
    assert hasattr(mlp, 'c_proj'), "Missing c_proj (should match HF naming)"
    
    print("✓ Model architecture matches HuggingFace GPT2 structure!\n")


def test_weight_tying():
    """Test that input embeddings and output head share weights"""
    print("=" * 50)
    print("Test 3: Testing weight tying")
    print("=" * 50)
    
    config = GPT2Config()
    model = GPT2(config)
    
    # Check that embeddings and lm_head share the same weight tensor
    assert model.transformer.wte.weight is model.lm_head.weight, \
        "Input embeddings and output head should share weights!"
    
    print("✓ Weight tying verified!\n")


def print_model_summary():
    """Print a summary of the model structure"""
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    
    config = GPT2Config(n_xmer_layer=12, n_head=12, embd_dim=768)
    model = GPT2(config)
    
    print("\nModel structure:")
    print(model)
    
    print("\n" + "=" * 50)
    print("Parameter names (first 20):")
    print("=" * 50)
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < 20:
            print(f"{name:50s} {list(param.shape)}")
        else:
            print(f"... and {sum(1 for _ in model.named_parameters()) - 20} more parameters")
            break


if __name__ == "__main__":
    print("\n🧪 Testing GPT2 Implementation\n")
    
    test_model_creation()
    test_model_architecture()
    test_weight_tying()
    print_model_summary()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print("\nTo load pretrained weights, run:")
    print("  model = GPT2.from_pretrained('gpt2')")
    print("\nNote: This requires the 'transformers' package.")
    print("Install with: pip install transformers")
