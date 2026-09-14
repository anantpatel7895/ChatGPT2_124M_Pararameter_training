# GPT-2 124M - Faithful PyTorch Implementation

A faithful reproduction of OpenAI's GPT-2 (124M parameters) from scratch, matching HuggingFace's implementation.

## ✨ Features

- ✅ **100% Compatible** with HuggingFace GPT-2 weights
- ✅ **Verified Accuracy**: Max output difference < 0.0001
- ✅ **Clean Architecture**: Modular, readable, well-documented code
- ✅ **Pre-trained Weights**: Load official GPT-2 checkpoints
- ✅ **124M Parameters**: Full GPT-2 small model

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ChatGPT2_124M_Pararameter_training

# Activate virtual environment
pyenv activate venv_slm

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.xmer_arch.gpt2 import GPT2
from transformers import GPT2Tokenizer
import torch

# Load pretrained model
model = GPT2.from_pretrained('gpt2')
model.eval()

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    logits, _ = model(input_ids)
    next_token_id = torch.argmax(logits[0, -1, :])
    next_token = tokenizer.decode([next_token_id])
    
print(f"Next token: {next_token}")
```

## 📁 Project Structure

```
src/
├── config_models/
│   └── gpt2_config.py          # Model configuration
└── xmer_arch/
    ├── gpt2.py                 # Main GPT-2 model
    ├── xmer_layer.py           # Transformer block
    ├── multihead_attention.py  # Multi-head attention
    └── feed_forward_mlp.py     # Feed-forward network

tests/                          # Test and verification scripts
demo_generation.py              # Text generation demo
```

## 🏗️ Architecture

### Model Specifications (GPT-2 124M)
- **Vocabulary Size**: 50,257 tokens
- **Max Sequence Length**: 1,024 tokens
- **Embedding Dimension**: 768
- **Number of Layers**: 12
- **Attention Heads**: 12 (64 dimensions each)
- **MLP Hidden Size**: 3,072 (4x embedding)
- **Total Parameters**: 124,439,808

### Key Components

1. **Token & Position Embeddings**: Learned embeddings for tokens and positions
2. **Transformer Blocks** (12x):
   - Pre-LayerNorm architecture
   - Multi-head self-attention with causal masking
   - Feed-forward MLP with GELU activation
   - Residual connections
3. **Output Layer**: Language modeling head with weight tying

## 🧪 Testing & Verification

Run the comprehensive test suite:

```bash
# Basic model tests
python tests/test_gpt2.py

# Pretrained weight loading tests
python tests/test_pretrained.py

# Quick comparison with HuggingFace
python tests/quick_comparison.py
```

### Test Results
```
✅ Model creation and forward pass
✅ Parameter count: 124,439,808
✅ Weight loading from HuggingFace
✅ Output comparison: Max diff < 0.0001
✅ Predictions match exactly
```

## 🎯 Demo

Run the text generation demo:

```bash
python demo_generation.py
```

This will:
1. Load the pretrained GPT-2 model
2. Generate text from example prompts
3. Enter interactive mode for custom prompts

## 🔍 Implementation Details

### Key Technical Choices

1. **GELU Activation**: Uses `GELU(approximate='tanh')` to match HF's `NewGELUActivation`
2. **Weight Initialization**: 
   - Normal distribution (std=0.02) for Linear/Embedding
   - Scaled initialization for residual projections
3. **Weight Loading**: 
   - Transposes Conv1D weights to Linear format
   - Handles all HuggingFace checkpoint formats
4. **Causal Masking**: Lower triangular attention mask for autoregressive generation

### Differences from HuggingFace

- Uses `nn.Linear` instead of custom `Conv1D` (functionally equivalent after transpose)
- Slightly simplified code structure while maintaining exact functionality
- All layer names match HF for easy weight loading

## 📊 Verification

Comparison with HuggingFace GPT-2:

```python
Prompt: "The meaning of life is"

Top prediction (both models): " not"
Max absolute difference: 0.00006866
Mean absolute difference: 0.00000998

✅ Models produce virtually identical outputs!
```

## 🛠️ Development

### Requirements
- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- Pydantic 2.0+

### Running Tests
```bash
# All tests
python tests/test_gpt2.py
python tests/test_pretrained.py

# Layer-by-layer debugging
python tests/debug_layers.py

# Weight comparison
python tests/debug_weights.py
```

## 📚 References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📝 License

This project is for educational purposes. GPT-2 model weights are subject to OpenAI's license.

## 🙏 Acknowledgments

- OpenAI for the original GPT-2 model
- HuggingFace for the reference implementation
- The PyTorch team for the excellent framework

---

**Status**: ✅ Complete and Verified

For detailed implementation notes, see [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)