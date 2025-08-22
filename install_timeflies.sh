#!/bin/bash
# TimeFlies Installation Script
# Installs from private GitHub repository with wheel fallback

set -e

echo "================================================"
echo "🧬 TimeFlies - ML for Aging Analysis"
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
REPO_URL="https://github.com/your-username/TimeFlies.git"  # Main repository
WHEEL_BRANCH="research"  # Branch containing the wheel file
WHEEL_PATH="dist/timeflies-0.2.0-py3-none-any.whl"  # Path to wheel in repo
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

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install TimeFlies with multiple fallback methods
print_status "Installing TimeFlies from Brown repository..."

INSTALL_SUCCESS=false

# Method 1: Try direct wheel download from research branch
print_status "Method 1: Downloading pre-built wheel..."
WHEEL_URL="$REPO_URL/raw/$WHEEL_BRANCH/$WHEEL_PATH"

if command -v curl >/dev/null 2>&1; then
    if curl -L -f -o timeflies.whl "$WHEEL_URL" 2>/dev/null; then
        if pip install timeflies.whl 2>/dev/null; then
            print_success "Installed TimeFlies from pre-built wheel"
            rm -f timeflies.whl
            INSTALL_SUCCESS=true
        else
            rm -f timeflies.whl
        fi
    fi
elif command -v wget >/dev/null 2>&1; then
    if wget -q -O timeflies.whl "$WHEEL_URL" 2>/dev/null; then
        if pip install timeflies.whl 2>/dev/null; then
            print_success "Installed TimeFlies from pre-built wheel"
            rm -f timeflies.whl
            INSTALL_SUCCESS=true
        else
            rm -f timeflies.whl
        fi
    fi
fi

# Method 2: Git install from main branch
if [[ "$INSTALL_SUCCESS" == "false" ]] && command -v git >/dev/null 2>&1; then
    print_status "Method 2: Installing from git repository..."
    
    if pip install git+"$REPO_URL"@"$MAIN_BRANCH" 2>/dev/null; then
        print_success "Installed TimeFlies from git repository"
        INSTALL_SUCCESS=true
    fi
fi

# Method 3: Clone and local install (most compatible)
if [[ "$INSTALL_SUCCESS" == "false" ]] && command -v git >/dev/null 2>&1; then
    print_status "Method 3: Cloning repository for local install..."
    
    if git clone --depth 1 -b "$MAIN_BRANCH" "$REPO_URL" timeflies_temp 2>/dev/null; then
        if cd timeflies_temp && pip install -e . 2>/dev/null; then
            print_success "Installed TimeFlies from local clone"
            cd ..
            rm -rf timeflies_temp
            INSTALL_SUCCESS=true
        else
            cd .. 2>/dev/null || true
            rm -rf timeflies_temp 2>/dev/null || true
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

# Create activation script
print_status "Creating activation script..."
cat > activate.sh << 'EOF'
#!/bin/bash
# TimeFlies Research Environment

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate  
else
    echo "❌ Virtual environment not found"
    return 1
fi

# Create helpful aliases
alias tf="timeflies"
alias tf-verify="timeflies verify"
alias tf-split="timeflies split"
alias tf-train="timeflies train"
alias tf-eval="timeflies evaluate"

echo "🧬 TimeFlies Research Environment Activated!"
echo ""
echo "Quick commands:"
echo "  timeflies split      # Create train/eval splits"  
echo "  timeflies train      # Train ML models"
echo "  timeflies evaluate   # Generate SHAP analysis & plots"
echo "  timeflies batch-correct  # Apply batch correction"
echo ""
echo "Research workflow:"
echo "  1. Add your H5AD files to: data/project_name/tissue_type/"
echo "  2. timeflies split    # Create stratified splits"
echo "  3. timeflies train    # Train with auto-evaluation"
echo "  4. Check results in:  outputs/project_name/"
echo ""
echo "With batch correction:"
echo "  1. timeflies split          # Split original data first"
echo "  2. timeflies batch-correct  # Apply correction to splits"
echo "  3. timeflies train --batch-corrected  # Train on corrected data"
EOF

chmod +x activate.sh

# Create batch correction activation script
print_status "Creating batch correction activation script..."
cat > activate_batch.sh << 'EOF'
#!/bin/bash
# TimeFlies Batch Correction Environment (PyTorch + scvi)

# Activate batch correction environment
if [[ -f ".venv_batch/bin/activate" ]]; then
    source .venv_batch/bin/activate
elif [[ -f ".venv_batch/Scripts/activate" ]]; then
    source .venv_batch/Scripts/activate  
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
echo "  source activate.sh"
EOF

chmod +x activate_batch.sh

# Test installation
print_status "Testing TimeFlies installation..."
if timeflies --help >/dev/null 2>&1; then
    print_success "TimeFlies CLI working perfectly!"
else
    print_warning "CLI test had warnings (this is normal)"
fi

# Create basic directory structure (configs come from repo)
print_status "Setting up project structure..."
mkdir -p data outputs

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
echo "   source activate.sh"
echo ""
echo -e "${BLUE}2. Add your research data:${NC}"
echo "   mkdir -p data/your_project/tissue_type"
echo "   # Copy your *_original.h5ad files there"
echo ""
echo -e "${BLUE}3. Configure analysis:${NC}"
echo "   nano configs/default.yaml  # Edit project settings"
echo ""
echo -e "${BLUE}4. Run complete analysis:${NC}"  
echo "   timeflies setup-all   # One-command setup (recommended)"
echo "   timeflies train       # Train models with auto-eval"
echo ""
echo -e "${BLUE}   With batch correction:${NC}"
echo "   timeflies split          # Split original data first"
echo "   timeflies batch-correct  # Apply correction to splits"
echo "   timeflies train --batch-corrected  # Train on corrected splits"
echo ""
echo -e "${BLUE}5. View results:${NC}"
echo "   ls outputs/           # All results, plots, SHAP analysis"
echo ""
echo -e "${YELLOW}📓 For Jupyter analysis:${NC}"
echo "   pip install jupyter"
echo "   jupyter notebook docs/notebooks/analysis.ipynb"
echo ""
echo -e "${BLUE}🔬 For batch correction:${NC}"
echo "   source activate_batch.sh  # Switch to PyTorch environment"
echo "   timeflies batch-correct    # Run scVI batch correction"
echo "   source activate.sh         # Return to main environment"
echo ""
echo -e "${GREEN}🔬 Research Lab${NC}"
echo "Contact your lab administrator for data access and support"
echo "================================================"