#!/bin/bash
# Run tests with proper Python path

cd "$(dirname "$0")"

echo "🧪 Running GPT-2 Tests"
echo "=" | awk '{for(i=0;i<70;i++)printf"=";printf"\n"}'

# Activate virtual environment
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    pyenv activate venv_slm 2>/dev/null || true
fi

# Run tests
echo ""
echo "Test 1: Basic Model Test"
echo "-" | awk '{for(i=0;i<70;i++)printf"-";printf"\n"}'
PYTHONPATH=. python tests/test_gpt2.py

echo ""
echo ""
echo "Test 2: Pretrained Model Test"
echo "-" | awk '{for(i=0;i<70;i++)printf"-";printf"\n"}'
PYTHONPATH=. python tests/test_pretrained.py

echo ""
echo ""
echo "Test 3: Quick Comparison"
echo "-" | awk '{for(i=0;i<70;i++)printf"-";printf"\n"}'
PYTHONPATH=. python tests/quick_comparison.py

echo ""
echo "=" | awk '{for(i=0;i<70;i++)printf"=";printf"\n"}'
echo "✅ All tests complete!"
