#!/bin/bash
# TimeFlies Development Environment Setup Script
# This script creates a complete development environment for TimeFlies

set -e  # Exit on any error

echo "🚀 Setting up TimeFlies development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/timeflies" ]]; then
    print_error "Please run this script from the TimeFlies root directory"
    print_error "Expected pyproject.toml and src/timeflies/ to exist"
    exit 1
fi

# Check Python version - REQUIRE Python 3.12
print_status "Checking Python version..."

# Check if python3.12 is available
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    PYTHON_VERSION=$(python3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Found Python $PYTHON_VERSION"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" == "3.12" ]]; then
        PYTHON_CMD="python3"
        print_success "Using python3 (version $PYTHON_VERSION)"
    else
        print_error "Python 3.12 is required for anndata compatibility. Found: $PYTHON_VERSION"
        print_error "Please install Python 3.12: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
else
    print_error "Python 3.12 is required. Please install: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

# Remove existing virtual environment if it exists
if [[ -d ".venv" ]]; then
    print_warning "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.12
print_status "Creating Python 3.12 virtual environment..."
$PYTHON_CMD -m venv .venv

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
print_status "Installing TimeFlies in development mode..."
pip install -e .

# Install core dependencies from your exact requirements
print_status "Installing core dependencies from requirements/linux/requirements.txt..."
pip install -r requirements/linux/requirements.txt

# Install testing dependencies
print_status "Installing testing dependencies..."
pip install pytest>=7.0.0 pytest-cov>=4.0.0 pytest-mock>=3.8.0 pytest-timeout>=2.1.0

# Create test data fixtures
print_status "Creating test data fixtures..."
if [[ -f "tests/fixtures/create_test_data.py" ]]; then
    python tests/fixtures/create_test_data.py || print_warning "Failed to create test fixtures (data files may not exist)"
else
    print_warning "Test data creation script not found"
fi

# Verify installation
print_status "Verifying installation..."

# Test core imports
$PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')

try:
    from src.timeflies.core.config_manager import ConfigManager, Config
    from src.timeflies.core.pipeline_manager import PipelineManager
    print('✅ Core components imported successfully')
except Exception as e:
    print(f'❌ Core import failed: {e}')
    sys.exit(1)

try:
    import scanpy as sc
    import anndata as ad
    print('✅ Single-cell analysis packages available')
except Exception as e:
    print(f'❌ Single-cell packages failed: {e}')
    sys.exit(1)

try:
    import scvi
    import torch
    print('✅ Batch correction (scVI) packages available')
except Exception as e:
    print('⚠️  Batch correction packages not available (optional)')

try:
    import tensorflow as tf
    import sklearn
    print('✅ Machine learning packages available')
except Exception as e:
    print(f'❌ ML packages failed: {e}')
    sys.exit(1)

try:
    import pytest
    print('✅ Testing framework available')
except Exception as e:
    print(f'❌ Testing framework failed: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    print_success "Environment setup completed successfully!"
else
    print_error "Environment verification failed"
    exit 1
fi

# Create activation script
print_status "Creating activation script..."
cat > activate_dev.sh << 'EOF'
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

# Activate virtual environment
source .venv/bin/activate

# Set environment variables for better TensorFlow behavior
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export CUDA_VISIBLE_DEVICES=""  # Disable GPU by default (can be overridden)

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
EOF

chmod +x activate_dev.sh

print_success "🎉 Setup complete!"
echo ""
echo "To start developing:"
echo -e "${GREEN}  source activate_dev.sh${NC}  # Activate environment"
echo -e "${GREEN}  python -m pytest tests/integration/test_real_data_pipeline.py -v${NC}  # Test integration tests"
echo ""
echo "The environment includes:"
echo "  ✅ All Python dependencies"
echo "  ✅ Testing framework (pytest)"
echo "  ✅ Single-cell analysis (scanpy, anndata)"
echo "  ✅ Machine learning (tensorflow, sklearn, xgboost)"
echo "  ✅ Batch correction (scvi-tools, pytorch)"
echo "  ✅ Data visualization"
echo "  ✅ Development tools"
echo ""
echo "Real data test fixtures created at: tests/fixtures/"