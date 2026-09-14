#!/usr/bin/env python3
"""Download Llama-7B model from Hugging Face."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import config
from src.core.logging import logger

# Load environment variables
load_dotenv("./environments/.env.dev")


def check_token():
    """Check if HuggingFace token is available."""
    token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    if not token:
        logger.error("❌ HUGGINGFACE_TOKEN not found in environment variables")
        logger.info("Please set your token:")
        logger.info("1. Get token from: https://huggingface.co/settings/tokens")
        logger.info("2. Accept Llama-2 license: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        logger.info("3. Set in .env file: HUGGINGFACE_TOKEN=your_token_here")
        sys.exit(1)
    return token


def download_model(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    save_dir: str = "models/base",
    cache_dir: str = ".cache/huggingface"
):
    """Download model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        save_dir: Directory to save the model
        cache_dir: Cache directory for downloads
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        logger.error("❌ Required packages not installed")
        logger.info("Run: pip install -r requirements/base.txt")
        sys.exit(1)
    
    # Get token
    token = check_token()
    
    # Create directories
    save_path = Path(save_dir)
    cache_path = Path(cache_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting download for: {model_name}")
    logger.info(f"📁 Save directory: {save_path.absolute()}")
    logger.info(f"📦 Cache directory: {cache_path.absolute()}")
    
    # Set cache environment variables
    os.environ['HF_HOME'] = str(cache_path)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
    
    try:
        # Download tokenizer
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            cache_dir=str(cache_path)
        )
        tokenizer.save_pretrained(str(save_path))
        logger.info(f"✅ Tokenizer saved to {save_path}")
        
        # Download model
        logger.info("📥 Downloading model (this may take 10-20 minutes)...")
        logger.info("⏳ Model size: ~13GB")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            cache_dir=str(cache_path),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None  # Don't load to device yet
        )
        model.save_pretrained(str(save_path))
        logger.info(f"✅ Model saved to {save_path}")
        
        # Check downloaded files
        model_files = list(save_path.glob("*"))
        logger.info(f"📄 Downloaded {len(model_files)} files")
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in model_files if f.is_file())
        size_gb = total_size / (1024 ** 3)
        logger.info(f"💾 Total size: {size_gb:.2f} GB")
        
        # Clean up cache directory
        logger.info("🧹 Cleaning up cache directory...")
        try:
            import shutil
            if cache_path.exists():
                # Calculate cache size before deletion
                cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                cache_size_gb = cache_size / (1024 ** 3)
                
                # Delete cache
                shutil.rmtree(cache_path)
                logger.info(f"✅ Cleaned up {cache_size_gb:.2f} GB from cache")
                logger.info(f"   Cache directory removed: {cache_path}")
            else:
                logger.info("   No cache to clean")
        except Exception as cache_err:
            logger.warning(f"⚠️  Failed to clean cache (non-critical): {cache_err}")
            logger.info("   You can manually delete: {cache_path}")
        
        logger.info("✅ Download complete!")
        logger.info(f"📁 Model available at: {save_path.absolute()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Verify your HuggingFace token is valid")
        logger.info("2. Check you've accepted the Llama-2 license")
        logger.info("3. Ensure you have ~15GB free disk space")
        logger.info("4. Check your internet connection")
        return False


def download_alternative_models():
    """Download alternative smaller models for testing."""
    alternatives = [
        ("microsoft/phi-2", "models/base/phi-2"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "models/base/tinyllama"),
    ]
    
    logger.info("\n🔄 Alternative models available:")
    for idx, (model, path) in enumerate(alternatives, 1):
        logger.info(f"{idx}. {model} (smaller, faster)")
    
    choice = input("\nDownload alternative? (y/n): ").lower()
    if choice == 'y':
        idx = int(input("Enter model number: ")) - 1
        model_name, save_dir = alternatives[idx]
        return download_model(model_name, save_dir)
    return False


if __name__ == "__main__":
    import argparse

    """

    ## Recommended: Start with Smaller Model

    For faster testing, start with **TinyLlama**:

    ```bash
    python scripts/download_model.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --save-dir "models/base/tinyllama"
    ```
    """


    
    parser = argparse.ArgumentParser(description="Download Llama-7B model")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--save-dir",
        default="models/base",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/huggingface",
        help="Cache directory"
    )
    parser.add_argument(
        "--alternative",
        action="store_true",
        help="Show alternative models"
    )
    
    args = parser.parse_args()
    
    if args.alternative:
        download_alternative_models()
    else:
        success = download_model(args.model, args.save_dir, args.cache_dir)
        sys.exit(0 if success else 1)

    
