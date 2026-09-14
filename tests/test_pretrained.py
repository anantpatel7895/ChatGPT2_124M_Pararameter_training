"""
Test script to verify loading pretrained GPT2 weights from HuggingFace
"""
import torch
import sys; sys.path.append("..")
from src.xmer_arch.gpt2 import GPT2


def test_pretrained_loading():
    """Test loading pretrained GPT2 model"""
    print("=" * 70)
    print("Testing GPT2.from_pretrained()")
    print("=" * 70)
    
    try:
        print("\nLoading pretrained GPT2 (124M parameters) from HuggingFace...")
        print("This may take a few minutes on first run...\n")
        
        model = GPT2.from_pretrained('gpt2')
        
        print("\n✓ Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        
        # Test inference
        print("\n" + "=" * 70)
        print("Testing inference with pretrained model")
        print("=" * 70)
        
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Test prompt
        prompt = "Hello, I am a language model"
        print(f"\nInput: '{prompt}'")
        
        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        print(f"Input tokens shape: {input_ids.shape}")
        
        # Generate
        model.eval()
        with torch.no_grad():
            logits, _ = model(input_ids)
        
        print(f"Output logits shape: {logits.shape}")
        
        # Get next token prediction
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode([next_token_id])
        
        print(f"Next predicted token: '{next_token}'")
        
        print("\n✓ Inference successful!")
        
        return model
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install the transformers library:")
        print("  pip install transformers")
        return None


def compare_with_huggingface():
    """Compare outputs with HuggingFace implementation"""
    print("\n" + "=" * 70)
    print("Comparing with HuggingFace GPT2LMHeadModel")
    print("=" * 70)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Load both models
        print("\nLoading custom GPT2...")
        custom_model = GPT2.from_pretrained('gpt2')
        
        print("Loading HuggingFace GPT2...")
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Prepare input
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        prompt = "The quick brown fox"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Get outputs
        custom_model.eval()
        hf_model.eval()
        
        with torch.no_grad():
            custom_logits, _ = custom_model(input_ids)
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits
        
        # Compare
        print(f"\nCustom model logits shape: {custom_logits.shape}")
        print(f"HuggingFace logits shape: {hf_logits.shape}")
        
        # Check if outputs match
        max_diff = torch.max(torch.abs(custom_logits - hf_logits)).item()
        mean_diff = torch.mean(torch.abs(custom_logits - hf_logits)).item()
        
        print(f"\nMax absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print("\n✅ Outputs match perfectly! 🎉")
        elif max_diff < 1e-2:
            print("\n✅ Outputs are very close (acceptable difference)")
        else:
            print("\n⚠️  Warning: Outputs differ significantly")
            print("This might indicate an architecture mismatch")
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install transformers library")


if __name__ == "__main__":
    print("\n🧪 Testing Pretrained GPT2 Loading\n")
    
    model = test_pretrained_loading()
    
    if model is not None:
        compare_with_huggingface()
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)
