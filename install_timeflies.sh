#!/bin/bash
# TimeFlies Installation Script
# Installs from private GitHub repository with wheel fallback

set -e

echo "================================================"
echo "TimeFlies v1.0 - ML for Aging Analysis"
echo "================================================"
echo "Machine learning for aging analysis in Drosophila"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}►${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration - Update these URLs for your repository
REPO_URL="git@github.com:rsinghlab/TimeFlies.git"  # Main repository
MAIN_BRANCH="main"  # Main development branch

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then echo "windows"
    else echo "unknown"; fi
}

PLATFORM=$(detect_os)
print_status "Platform: $PLATFORM"
print_status "Working directory: $(pwd)"

# Find Python 3.12+
PYTHON_CMD=""
for cmd in python3.12 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VERSION=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ $(echo "$VERSION >= 3.12" | python3 -c "
import sys
try:
    v1, v2 = '$VERSION', '3.12'
    print(1 if float(v1) >= float(v2) else 0)
except:
    print(0)
" 2>/dev/null || echo "0") == "1" ]]; then
            PYTHON_CMD="$cmd"
            print_success "Found Python $VERSION"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    print_error "Python 3.12+ required but not found"
    case $PLATFORM in
        linux) echo "Install: sudo apt install python3.12 python3.12-venv" ;;
        macos) echo "Install: brew install python@3.12" ;;
        windows) echo "Install from: https://www.python.org/downloads/" ;;
    esac
    exit 1
fi

# Create virtual environment
print_status "Creating Python environment..."
if [[ -d ".venv" ]]; then
    print_warning "Removing existing environment..."
    rm -rf .venv
fi

$PYTHON_CMD -m venv .venv

# Activate environment
if [[ "$PLATFORM" == "windows" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

print_success "Environment created and activated"

# Upgrade pip and ensure GUI dependencies
print_status "Upgrading pip and installing GUI dependencies..."
pip install --upgrade pip

# Install tkinter if needed (for GUI support)
if ! python -c "import tkinter" >/dev/null 2>&1; then
    print_status "Installing GUI support (tkinter)..."
    case $PLATFORM in
        linux)
            if command -v apt >/dev/null 2>&1; then
                sudo apt update && sudo apt install -y python3-tk || print_warning "Could not auto-install tkinter. Install manually: sudo apt install python3-tk"
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y tkinter || print_warning "Could not auto-install tkinter. Install manually: sudo yum install tkinter"
            else
                print_warning "Please install tkinter manually for GUI support"
            fi
            ;;
        macos)
            print_warning "If GUI doesn't work, install tkinter via: brew install python-tk"
            ;;
        windows)
            print_warning "tkinter should be included with Python on Windows"
            ;;
    esac
else
    print_success "GUI support (tkinter) already available"
fi

# Install TimeFlies
print_status "Installing TimeFlies..."

INSTALL_SUCCESS=false

if command -v git >/dev/null 2>&1; then
    print_status "Downloading and installing TimeFlies..."

    # Clone to permanent directory for user install (needed for editable install)
    if git clone --depth 1 -b "$MAIN_BRANCH" "$REPO_URL" .timeflies_src >/dev/null 2>&1; then
        print_status "Installing dependencies (this may take a few minutes)..."
        if cd .timeflies_src && pip install -e . >/dev/null 2>&1; then
            print_success "Installed TimeFlies successfully"
            cd ..
            # Keep the source directory for editable install
            INSTALL_SUCCESS=true
        else
            cd .. 2>/dev/null || true
            rm -rf .timeflies_src 2>/dev/null || true
        fi
    fi
fi

if [[ "$INSTALL_SUCCESS" == "false" ]]; then
    print_error "Failed to install TimeFlies"
    echo ""
    print_error "Please check:"
    print_error "1. You have access to the repository: $REPO_URL"
    print_error "2. Your GitHub authentication is set up"
    print_error "3. Internet connection is working"
    echo ""
    print_status "To set up GitHub access:"
    echo "   1. Generate a personal access token at https://github.com/settings/tokens"
    echo "   2. Use: git config --global credential.helper store"
    echo "   3. Try: git clone $REPO_URL (enter token when prompted)"
    exit 1
fi

print_success "TimeFlies installed successfully!"

# Create activation script (hidden)
print_status "Creating activation script..."
cat > .activate.sh << 'EOF'
#!/bin/bash
# TimeFlies Research Environment

# Suppress TensorFlow/CUDA warnings and logs for cleaner output
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export GRPC_VERBOSITY=ERROR
export AUTOGRAPH_VERBOSITY=0
export CUDA_VISIBLE_DEVICES=""
export ABSL_LOG_LEVEL=ERROR

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    # Clean up prompt - remove any existing venv indicators
    PS1="${PS1//(.venv) /}"
    PS1="${PS1//(.venv)/}"
    PS1="${PS1//((.venv) )/}"
    PS1="${PS1//((.venv))/}"
    PS1="${PS1//() /}"
    PS1="${PS1//()/}"
    export PS1="(.venv) ${PS1}"
elif [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate
    # Clean up prompt - remove any existing venv indicators
    PS1="${PS1//(.venv) /}"
    PS1="${PS1//(.venv)/}"
    PS1="${PS1//((.venv) )/}"
    PS1="${PS1//((.venv))/}"
    PS1="${PS1//() /}"
    PS1="${PS1//()/}"
    export PS1="(.venv) ${PS1}"
else
    echo "❌ Virtual environment not found"
    return 1
fi

# Create helpful aliases
alias tf="timeflies"
alias tf-setup="timeflies setup"
alias tf-verify="timeflies verify"
alias tf-split="timeflies split"
alias tf-train="timeflies train"
alias tf-eval="timeflies evaluate"
alias tf-analyze="timeflies analyze"
alias tf-eda="timeflies eda"
alias tf-tune="timeflies tune"
alias tf-queue="timeflies queue"
alias tf-test="timeflies test"

echo "🧬 TimeFlies Research Environment Activated!"
echo ""
echo "Research workflow:"
echo "  timeflies setup              # Complete setup workflow"
echo "  timeflies train              # Train ML models with evaluation"
echo "  timeflies evaluate           # Evaluate models on test data"
echo "  timeflies analyze            # Project-specific analysis scripts"
echo ""
echo "Quick commands:"
echo "  timeflies split              # Create train/eval splits"
echo "  timeflies batch-correct      # Apply batch correction"
echo "  timeflies verify             # System verification"
echo "  timeflies eda                # Exploratory data analysis"
echo "  timeflies tune               # Hyperparameter tuning"
echo "  timeflies queue              # Model queue training"
echo "  timeflies test               # Run tests"
echo "  timeflies update             # Update TimeFlies"
echo ""
echo "Getting started:"
echo "  1. Add your H5AD files to: data/project_name/tissue_type/"
echo "  2. timeflies setup           # Setup and verify everything"
echo "  3. timeflies train           # Train with auto-evaluation"
echo "  4. Check results in:         outputs/project_name/"
EOF

chmod +x .activate.sh

# Create batch correction activation script (hidden)
print_status "Creating batch correction activation script..."
cat > .activate_batch.sh << 'EOF'
#!/bin/bash
# TimeFlies Batch Correction Environment (PyTorch + scvi)

# Suppress TensorFlow/CUDA warnings and logs for cleaner output
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export GRPC_VERBOSITY=ERROR
export AUTOGRAPH_VERBOSITY=0
export CUDA_VISIBLE_DEVICES=""
export ABSL_LOG_LEVEL=ERROR

# Activate batch correction environment
if [[ -f ".venv_batch/bin/activate" ]]; then
    source .venv_batch/bin/activate
    # Clean up prompt - remove any existing (.venv_batch) and empty parentheses
    PS1="\${PS1//\\(.venv_batch\\) /}"
    PS1="\${PS1//\\(.venv_batch\\)/}"
    PS1="\${PS1//\\(\\) /}"
    PS1="\${PS1//\\(\\)/}"
    export PS1="(.venv_batch) \${PS1}"
elif [[ -f ".venv_batch/Scripts/activate" ]]; then
    source .venv_batch/Scripts/activate
    # Clean up prompt - remove any existing (.venv_batch) and empty parentheses
    PS1="\${PS1//\\(.venv_batch\\) /}"
    PS1="\${PS1//\\(.venv_batch\\)/}"
    PS1="\${PS1//\\(\\) /}"
    PS1="\${PS1//\\(\\)/}"
    export PS1="(.venv_batch) \${PS1}"
else
    echo "❌ Batch correction environment not found"
    return 1
fi

echo "🧬 TimeFlies Batch Correction Environment Activated!"
echo ""
echo "Available tools:"
echo "  timeflies batch-correct  # Run scVI batch correction"
echo "  python                   # PyTorch + scvi-tools available"
echo ""
echo "To return to main environment:"
echo "  deactivate"
echo "  source .activate.sh"
EOF

chmod +x .activate_batch.sh

# Set Windows hidden attribute for WSL
if grep -q microsoft /proc/version 2>/dev/null; then
    attrib +h .venv .activate.sh .activate_batch.sh .timeflies_src .venv_batch 2>/dev/null || true
fi

# Test installation
print_status "Testing TimeFlies installation..."
if timeflies --help >/dev/null 2>&1; then
    print_success "TimeFlies CLI working perfectly!"
else
    print_warning "CLI test had warnings (this is normal)"
fi

# Create basic directory structure
print_status "Setting up project structure..."
mkdir -p data

# Basic project structure (user mode only)

# Setup batch correction environment
print_status "Setting up batch correction environment..."
if [[ ! -d ".venv_batch" ]]; then
    print_status "Creating PyTorch environment for batch correction..."
    $PYTHON_CMD -m venv .venv_batch

    # Activate batch environment
    if [[ "$PLATFORM" == "windows" ]]; then
        source .venv_batch/Scripts/activate
    else
        source .venv_batch/bin/activate
    fi

    # Install PyTorch + scvi-tools
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install scvi-tools scanpy pandas numpy matplotlib seaborn

    # Deactivate batch environment
    deactivate

    # Reactivate main environment
    if [[ "$PLATFORM" == "windows" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi

    print_success "Batch correction environment created!"
else
    print_success "Batch correction environment already exists"
fi

print_success "================================================"
print_success "🎉 TimeFlies Installation Complete!"
print_success "================================================"
echo ""

echo -e "${GREEN}🔬 Ready for Research!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo -e "${BLUE}1. Activate environment:${NC}"
echo "   source .activate.sh"
echo ""
echo -e "${BLUE}2. Add your research data:${NC}"
echo "   mkdir -p data/your_project/tissue_type"
echo "   # Copy your *_original.h5ad files there"
echo ""
echo -e "${BLUE}3. Complete setup:${NC}"
echo "   timeflies setup              # Creates config, templates, outputs, GUI launcher"
echo ""
echo -e "${BLUE}4. Run analysis workflow:${NC}"
echo "   timeflies train              # Train models with auto-evaluation"
echo "   timeflies evaluate           # Evaluate models on test data"
echo "   timeflies analyze            # Project-specific analysis scripts"
echo ""
echo -e "${BLUE}5. View results:${NC}"
echo "   ls outputs/           # All results, plots, model analysis"
echo ""
echo -e "${YELLOW}📓 For Jupyter analysis:${NC}"
echo "   pip install jupyter"
echo "   jupyter notebook docs/notebooks/analysis.ipynb"
echo ""
echo -e "${GREEN}🖥️ For GUI Interface (after setup):${NC}"
echo "   python TimeFlies_Launcher.py  # Graphical interface"
echo ""
echo -e "${GREEN}🛠️ For Developers:${NC}"
echo "   git clone https://github.com/rsinghlab/TimeFlies.git"
echo "   cd TimeFlies && timeflies setup --dev  # Create environments"

echo ""
echo -e "${GREEN}🔄 Updates:${NC}"
echo "   timeflies update      # Update to latest version"
echo ""
echo -e "${GREEN}🔬 Research Lab${NC}"
echo "Contact your lab administrator for data access and support"
echo "================================================"
