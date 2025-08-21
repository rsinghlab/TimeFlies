#!/bin/bash
# TimeFlies Development Environment Setup
# Cross-platform setup for Linux, macOS, and Windows (WSL/Git Bash)
# Uses pyproject.toml for all dependencies

set -e  # Exit on any error

echo "================================================"
echo "TimeFlies Development Environment Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() { echo -e "${BLUE}►${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        case "$(uname -s)" in
            Linux*)     echo "linux";;
            Darwin*)    echo "macos";;
            CYGWIN*|MINGW*|MSYS*) echo "windows";;
            *)          echo "unknown";;
        esac
    fi
}

# Detect platform
PLATFORM=$(detect_os)
print_status "Detected platform: $PLATFORM"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/projects" ]] || [[ ! -d "src/shared" ]]; then
    print_error "Please run this script from the TimeFlies root directory"
    exit 1
fi

# Find Python (3.12+ required)
print_status "Looking for Python 3.12+..."

find_python() {
    # Try Python 3.12 first (newest)
    for py_cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v $py_cmd &> /dev/null; then
            version=$($py_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            major=${version%%.*}
            minor=${version#*.}
            
            if [[ $major -eq 3 ]] && [[ $minor -ge 12 ]]; then
                PYTHON_CMD=$py_cmd
                PYTHON_VERSION=$version
                return 0
            fi
        fi
    done
    
    # Special check for macOS Homebrew Python
    if [[ "$PLATFORM" == "macos" ]]; then
        for py_path in /opt/homebrew/bin/python3* /usr/local/bin/python3*; do
            if [[ -x "$py_path" ]]; then
                version=$($py_path -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
                major=${version%%.*}
                minor=${version#*.}
                
                if [[ $major -eq 3 ]] && [[ $minor -ge 12 ]]; then
                    PYTHON_CMD=$py_path
                    PYTHON_VERSION=$version
                    return 0
                fi
            fi
        done
    fi
    
    return 1
}

if ! find_python; then
    print_error "Python 3.12+ not found!"
    echo ""
    case "$PLATFORM" in
        linux)
            echo "Install Python with:"
            echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv python3-pip"
            echo "  Fedora: sudo dnf install python3.12"
            echo "  Arch: sudo pacman -S python"
            ;;
        macos)
            echo "Install Python with:"
            echo "  brew install python@3.12"
            echo "  Or download from: https://www.python.org/downloads/"
            ;;
        windows)
            echo "Install Python from:"
            echo "  https://www.python.org/downloads/windows/"
            echo "  Or: winget install Python.Python.3.12"
            ;;
    esac
    exit 1
fi

print_success "Found Python $PYTHON_VERSION at $PYTHON_CMD"

# Remove old virtual environment
if [[ -d ".venv" ]]; then
    print_warning "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment
print_success "Creating virtual environment..."
$PYTHON_CMD -m venv .venv

# Determine activation script path
if [[ "$PLATFORM" == "windows" ]]; then
    ACTIVATE_SCRIPT=".venv/Scripts/activate"
    PIP_CMD=".venv/Scripts/pip"
    PYTHON_VENV=".venv/Scripts/python"
else
    ACTIVATE_SCRIPT=".venv/bin/activate"
    PIP_CMD=".venv/bin/pip"
    PYTHON_VENV=".venv/bin/python"
    
    # Fix PS1 double parentheses issue
    if [[ -f "$ACTIVATE_SCRIPT" ]]; then
        sed -i.bak 's/PS1="("\x27(.venv) \x27") \${PS1:-}"/PS1="(.venv) \${PS1:-}"/' "$ACTIVATE_SCRIPT" 2>/dev/null || true
        rm -f "${ACTIVATE_SCRIPT}.bak" 2>/dev/null || true
    fi
fi

# Activate virtual environment
print_success "Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

# Upgrade pip
print_success "Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel --quiet

# Platform-specific pre-installation
case "$PLATFORM" in
    macos)
        print_status "Configuring for macOS..."
        export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
        export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
        
        # Check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            print_status "Detected Apple Silicon (M1/M2/M3)"
            # TensorFlow for Apple Silicon
            $PIP_CMD install tensorflow-macos --quiet 2>/dev/null || true
            $PIP_CMD install tensorflow-metal --quiet 2>/dev/null || true
        fi
        ;;
    windows)
        print_status "Configuring for Windows..."
        # Windows might need specific settings
        export PYTHONUTF8=1
        ;;
    linux)
        print_status "Configuring for Linux..."
        # Install GPU-enabled TensorFlow for Linux
        if command -v nvidia-smi &> /dev/null; then
            print_status "NVIDIA GPU detected, installing TensorFlow with CUDA support..."
            $PIP_CMD install tensorflow[and-cuda] --quiet 2>/dev/null || {
                print_warning "Failed to install tensorflow[and-cuda], falling back to regular tensorflow"
            }
        fi
        ;;
esac

# Install TimeFlies and dependencies from pyproject.toml
print_success "Installing TimeFlies and core dependencies..."
$PIP_CMD install -e . --quiet

# Install development dependencies
print_success "Installing development tools..."
$PIP_CMD install -e ".[dev]" --quiet 2>/dev/null || print_warning "Some dev tools may not be available"

# Ask about creating batch correction environment
echo ""
read -p "Create separate batch correction environment (.venv_batch)? [y/N] " -n 1 -r
echo
CREATE_BATCH_ENV=false
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CREATE_BATCH_ENV=true
    print_status "Will create separate batch correction environment"
else
    print_status "Skipping batch correction environment"
fi

# Verify installation
print_status "Verifying installation..."

verification_passed=true
$PYTHON_VENV -c "$(cat <<'EOF'
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

# Core imports  
try:
    from src.shared.core.active_config import get_active_project, get_config_for_active_project
    print('OK Core configuration modules')
except Exception as e:
    print(f'ERROR Core config modules: {e}')
    sys.exit(1)

# Shared pipeline modules
try:
    from src.shared.core.pipeline_manager import PipelineManager
    print('OK Shared pipeline modules')
except Exception as e:
    print(f'ERROR Shared pipeline modules: {e}')
    sys.exit(1)

# Scientific packages
try:
    import numpy, pandas, scanpy, anndata
    print('OK Scientific packages')
except Exception as e:
    print(f'ERROR Scientific packages: {e}')
    sys.exit(1)

# ML packages
try:
    import tensorflow, sklearn
    print('OK Machine learning packages')
except Exception as e:
    print(f'ERROR ML packages: {e}')
    sys.exit(1)

# Optional packages
try:
    import scvi, torch
    print('OK Batch correction packages')
except:
    print('! Batch correction packages optional')

try:
    import pytest  
    print('OK Testing framework')
except:
    print('! Testing framework optional')
EOF
)" || verification_passed=false

if [[ "$verification_passed" == "false" ]]; then
    print_error "Some components failed to install"
    exit 1
fi

# Create simple activation script
cat > activate.sh << 'EOF'
#!/bin/bash
# TimeFlies Environment Activation

# Store original PS1 (before venv modifies it)
if [ -n "${PS1}" ]; then
    ORIG_PS1="${PS1}"
else
    ORIG_PS1="$ "
fi

# Find and activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "Error: Virtual environment not found. Run setup_dev_env.sh first."
    exit 1
fi

# Fix PS1 double parentheses issue - override after activation
export PS1="(.venv) ${ORIG_PS1}"

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# Show status
echo "================================================"
echo "TimeFlies Environment Activated"
echo "================================================"
echo ""
echo "Quick commands:"
echo "  Verify setup:  python run_timeflies.py verify"
echo "  Create tests fixtures:  python run_timeflies.py create-test-data"
echo "  Setup splits:  python run_timeflies.py setup"
echo "  Train model:   python run_timeflies.py train"
echo "  Run tests:     python run_timeflies.py test"
echo ""
echo "Workflow:"
echo "  1. Place data in data/[project]/[tissue]/"
echo "  2. Create test fixtures: python run_timeflies.py create-test-data"  
echo "  3. Setup train/eval splits: python run_timeflies.py setup"
echo "  4. Verify: python run_timeflies.py verify"
echo "  5. Train: python run_timeflies.py train"
echo "  6. Run tests: python run_timeflies.py test"
echo ""
echo "Python: $(which python) ($(python --version 2>&1))"
echo "================================================"
EOF

chmod +x activate.sh

# Create batch correction environment if requested
if [[ "$CREATE_BATCH_ENV" == "true" ]]; then
    echo ""
    print_status "================================================"
    print_status "Creating Batch Correction Environment"
    print_status "================================================"
    
    # Remove old batch environment
    if [[ -d ".venv_batch" ]]; then
        print_warning "Removing existing batch correction environment..."
        rm -rf .venv_batch
    fi
    
    # Create batch environment
    print_status "Creating batch correction virtual environment..."
    $PYTHON_CMD -m venv .venv_batch
    
    # Determine batch activation script path
    if [[ "$PLATFORM" == "windows" ]]; then
        BATCH_ACTIVATE_SCRIPT=".venv_batch/Scripts/activate"
        BATCH_PIP_CMD=".venv_batch/Scripts/pip"
        BATCH_PYTHON_VENV=".venv_batch/Scripts/python"
    else
        BATCH_ACTIVATE_SCRIPT=".venv_batch/bin/activate"
        BATCH_PIP_CMD=".venv_batch/bin/pip"
        BATCH_PYTHON_VENV=".venv_batch/bin/python"
    fi
    
    # Activate batch environment
    print_status "Activating batch correction environment..."
    source "$BATCH_ACTIVATE_SCRIPT"
    
    # Fix PS1 double parentheses issue for batch env
    if [[ -f "$BATCH_ACTIVATE_SCRIPT" ]]; then
        sed -i.bak 's/PS1="(\x27(.venv_batch) \x27") \${PS1:-}"/PS1="(.venv_batch) \${PS1:-}"/' "$BATCH_ACTIVATE_SCRIPT" 2>/dev/null || true
        rm -f "${BATCH_ACTIVATE_SCRIPT}.bak" 2>/dev/null || true
    fi
    
    # Upgrade pip in batch environment
    print_status "Upgrading pip in batch environment..."
    $BATCH_PIP_CMD install --upgrade pip setuptools wheel --quiet
    
    # Install PyTorch and batch correction tools
    print_status "Installing PyTorch and batch correction dependencies..."
    $BATCH_PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    $BATCH_PIP_CMD install scvi-tools scanpy anndata pandas numpy matplotlib seaborn --quiet
    
    # Install core TimeFlies for data access (without TensorFlow)
    print_status "Installing core TimeFlies modules for data access..."
    $BATCH_PIP_CMD install pyyaml dill --quiet
    
    # Create batch environment activation script
    cat > activate_batch.sh << 'EOF'
#!/bin/bash
# TimeFlies Batch Correction Environment Activation

# Find and activate batch virtual environment
if [ -f ".venv_batch/bin/activate" ]; then
    source .venv_batch/bin/activate
elif [ -f ".venv_batch/Scripts/activate" ]; then
    source .venv_batch/Scripts/activate
else
    echo "Error: Batch correction environment not found. Run setup_dev_env.sh first."
    exit 1
fi

# Show status
echo "================================================"
echo "TimeFlies Batch Correction Environment Activated"
echo "================================================"
echo ""
echo "Available tools:"
echo "  PyTorch:      Available"
echo "  scVI-tools:   Available"
echo ""
echo "Python: \$(which python) (\$(python --version 2>&1))"
echo "Project: \$(pwd)"
echo "================================================"
EOF

    chmod +x activate_batch.sh
    
    # Deactivate batch environment and reactivate main
    deactivate
    source "$ACTIVATE_SCRIPT"
    
    print_success "Batch correction environment created successfully!"
    print_status "Use: source activate_batch.sh to activate batch correction environment"
fi

# Success message
echo ""
print_success "================================================"
print_success "Setup Complete!"
print_success "================================================"
echo ""
echo "Platform:     $PLATFORM"
echo "Python:       $PYTHON_CMD ($PYTHON_VERSION)"
echo "Environment:  .venv/"
echo ""
echo "Environments created:"
echo "  Main (.venv):        TensorFlow + TimeFlies"
echo "  Batch (.venv_batch): PyTorch + scVI-tools (if created)"
echo ""
echo "COMPLETE WORKFLOW:"
echo "  1. Activate environment:     source activate.sh"
echo "  2. Place your data files in: data/[project]/[tissue]/"
echo "     - drosophila_head_aging_original.h5ad (required)"
echo "     - drosophila_head_alzheimers_original.h5ad (for alzheimer's project)"
echo "  3. Create test fixtures:     python run_timeflies.py create-test-data"
echo "  4. Setup data splits:        python run_timeflies.py setup"
if [[ "$CREATE_BATCH_ENV" == "true" ]]; then
echo "  5. Optional batch correction:"
echo "     a. Activate batch env:    source activate_batch.sh"
echo "     b. Batch correct splits:  python run_timeflies.py batch-correct"
echo "     c. Switch back to main:   source activate.sh"
echo "  6. Verify setup:             python run_timeflies.py verify"
echo "  7. Train models:             python run_timeflies.py train"
echo "     (alternatively use run_timeflies.py evaluate to evaluate models)"
else
echo "  5. Verify setup:             python run_timeflies.py verify"
echo "  6. Train models:             python run_timeflies.py train"
echo "     (Note: For batch correction, re-run setup and choose batch environment)"
fi
echo ""
echo "Data folder structure:"
echo "  data/fruitfly_aging/head/        - Aging project data"
echo "  data/fruitfly_alzheimers/head/   - Alzheimer's project data"  
echo "  tests/fixtures/[project]/        - Generated test data"
echo ""
print_success "Ready to go!"