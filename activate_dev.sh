#!/bin/bash
# TimeFlies Development Environment Activation Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧬 Activating TimeFlies Development Environment${NC}"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/timeflies" ]]; then
    echo "❌ Please run this script from the TimeFlies root directory"
    exit 1
fi

# Check if virtual environment is already activated
if [[ "$VIRTUAL_ENV" != *"TimeFlies/.venv"* ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment already activated"
fi

# Set environment variables for better TensorFlow behavior
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
# Note: CUDA_VISIBLE_DEVICES left unset to allow GPU usage

# Unset CUDA_VISIBLE_DEVICES in case it was set to empty string elsewhere
if [[ -n "${CUDA_VISIBLE_DEVICES+set}" ]] && [[ "$CUDA_VISIBLE_DEVICES" == "" ]]; then
    unset CUDA_VISIBLE_DEVICES
    echo "Cleared empty CUDA_VISIBLE_DEVICES to enable GPU access"
fi

echo -e "${GREEN}✅ Development environment activated${NC}"
echo ""
echo "Available commands:"
echo "  🧪 Run unit tests:        python -m pytest tests/unit/ -v"
echo "  🔗 Run integration tests: python -m pytest tests/integration/ -v"  
echo "  🚀 Run all tests:         python -m pytest tests/ -v"
echo "  🏃 Run specific test:     python -m pytest tests/integration/test_real_data_pipeline.py -v"
echo "  📊 Run with coverage:     python -m pytest tests/ --cov=src --cov-report=html"
echo "  🎯 Run TimeFlies CLI:     python -m timeflies.cli.main --help"
echo "  📝 Format code:           black src/ tests/"
echo "  🔍 Lint code:             flake8 src/ tests/"
echo ""
echo "To run the main pipeline:"
echo "  python train.py                    # Training"
echo "  python -m timeflies.cli.main train # CLI training"
echo ""
echo "To deactivate: deactivate"
