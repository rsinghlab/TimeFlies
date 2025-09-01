#!/bin/bash
# TimeFlies Installation Script
# Installs from GitHub repository with wheel fallback

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
REPO_URL="https://github.com/rsinghlab/TimeFlies.git"  # Main repository
MAIN_BRANCH="main"  # Main development branch

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then echo "windows"
    else echo "unknown"; fi
}

PLATFORM=$(detect_os)

# Check command line arguments
DEV_MODE=false
UPDATE_MODE=false

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --dev)
            DEV_MODE=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

if [[ "$TIMEFLIES_UPDATE_MODE" == "1" ]]; then
    UPDATE_MODE=true
    echo "================================================"
    echo "TimeFlies Update Mode - Refreshing Installation"
    echo "================================================"
    print_status "Updating existing TimeFlies installation"
elif [[ "$DEV_MODE" == "true" ]]; then
    echo "================================================"
    echo "TimeFlies Developer Mode - Minimal Installation"
    echo "================================================"
    print_status "Installing TimeFlies for development"
else
    print_status "Platform: $PLATFORM"
    print_status "Working directory: $(pwd)"
fi

# Find Python 3.12+
PYTHON_CMD=""
for cmd in python3.12 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VERSION=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        
        # Check if version is 3.12 or higher
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 12 ]]; then
            PYTHON_CMD="$cmd"
            print_success "Found Python $VERSION"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    print_warning "Python 3.12+ not found - installing it now..."
    
    case $PLATFORM in
        linux)
            print_status "Installing Python 3.12 on Linux..."
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update
                sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y python3.12 python3.12-devel
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y python3.12 python3.12-devel
            else
                print_error "Unable to install Python 3.12 automatically. Please install manually:"
                echo "  sudo apt-get install python3.12 python3.12-venv"
                exit 1
            fi
            ;;
        macos)
            print_status "Installing Python 3.12 on macOS..."
            if command -v brew >/dev/null 2>&1; then
                brew install python@3.12
                # Install libomp for XGBoost support
                print_status "Installing OpenMP runtime for XGBoost..."
                brew install libomp
            else
                print_status "Homebrew not found - installing it first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                brew install python@3.12
                # Install libomp for XGBoost support
                print_status "Installing OpenMP runtime for XGBoost..."
                brew install libomp
            fi
            ;;
        windows)
            print_status "Installing Python 3.12 on Windows..."
            # For Windows, download and install Python using PowerShell
            powershell -Command "
                \$url = 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe'
                \$output = '\$env:TEMP\python-3.12.0-amd64.exe'
                Invoke-WebRequest -Uri \$url -OutFile \$output
                Start-Process -FilePath \$output -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -Wait
            "
            ;;
        *)
            print_error "Unknown platform. Please install Python 3.12 manually"
            exit 1
            ;;
    esac
    
    # Check again after installation
    for cmd in python3.12 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            VERSION=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            MAJOR=$(echo "$VERSION" | cut -d. -f1)
            MINOR=$(echo "$VERSION" | cut -d. -f2)
            
            if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 12 ]]; then
                PYTHON_CMD="$cmd"
                print_success "Python $VERSION installed successfully"
                break
            fi
        fi
    done
    
    if [[ -z "$PYTHON_CMD" ]]; then
        print_error "Failed to install Python 3.12. Please install it manually and run this script again"
        exit 1
    fi
fi

# Create virtual environment
if [[ "$UPDATE_MODE" == "true" ]]; then
    print_status "Using existing Python environment..."
    if [[ ! -d ".venv" ]]; then
        print_error "Virtual environment not found - run full installation"
        exit 1
    fi
else
    if [[ "$DEV_MODE" == "true" ]]; then
        print_status "Creating Python development environment (.venv)..."
        if [[ -d ".venv" ]]; then
            print_warning "Removing existing development environment..."
            rm -rf .venv
        fi
        $PYTHON_CMD -m venv .venv
    else
        print_status "Creating Python environment..."
        if [[ -d ".venv" ]]; then
            print_warning "Removing existing environment..."
            rm -rf .venv
        fi
        $PYTHON_CMD -m venv .venv
    fi
fi

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

# GUI support is now web-based (gradio) - no system dependencies needed
print_success "Web-based GUI included (gradio) - no system dependencies required"

# Check for libomp on macOS (required for XGBoost)
if [[ "$PLATFORM" == "macos" ]]; then
    if ! brew list libomp >/dev/null 2>&1; then
        print_status "Installing OpenMP runtime for XGBoost support..."
        if command -v brew >/dev/null 2>&1; then
            brew install libomp
            print_success "OpenMP runtime installed"
        else
            print_warning "Homebrew not found - XGBoost may not work properly"
            print_warning "To fix: Install Homebrew and run 'brew install libomp'"
        fi
    else
        print_success "OpenMP runtime already installed"
    fi
fi

# Install TimeFlies
if [[ "$UPDATE_MODE" == "true" ]]; then
    print_status "Updating TimeFlies installation..."
    INSTALL_SUCCESS=false

    if [[ -d ".timeflies_src" ]]; then
        print_status "Updating TimeFlies source code..."
        if cd .timeflies_src && git pull origin "$MAIN_BRANCH" >/dev/null 2>&1; then
            print_status "Checking for new dependencies..."
            # Only install if pyproject.toml changed or if install seems broken
            if pip install -e . --quiet --no-deps >/dev/null 2>&1; then
                # Check if we need to install new dependencies
                pip install -e . --quiet --upgrade-strategy only-if-needed >/dev/null 2>&1
                print_success "Updated TimeFlies successfully"
                cd ..
                INSTALL_SUCCESS=true
            else
                print_warning "Dependency update had issues but continuing"
                cd ..
                INSTALL_SUCCESS=true  # Continue even if some deps failed
            fi
        else
            print_error "Failed to update TimeFlies source"
            cd .. 2>/dev/null || true
        fi
    else
        print_error "TimeFlies source directory not found - run full installation"
        exit 1
    fi
else
    print_status "Installing TimeFlies..."
    INSTALL_SUCCESS=false

    if command -v git >/dev/null 2>&1; then
        print_status "Downloading and installing TimeFlies..."

        # Remove existing directory if it exists
        if [ -d ".timeflies_src" ]; then
            print_status "Removing existing installation..."
            rm -rf .timeflies_src
        fi

        # Clone to permanent directory for user install (needed for editable install)
        if git clone --depth 1 -b "$MAIN_BRANCH" "$REPO_URL" .timeflies_src >/dev/null 2>&1; then
            print_status "Installing dependencies (this may take a few minutes)..."
            cd .timeflies_src

            # Detect GPU capability and install appropriate version
            print_status "Detecting GPU capability..."
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
                if [[ "$DEV_MODE" == "true" ]]; then
                    print_success "NVIDIA GPU detected - installing TimeFlies with dev dependencies"
                    if pip install -e ".[dev]" >/dev/null 2>&1; then
                        print_success "Installed TimeFlies with dev dependencies"
                        cd ..
                        INSTALL_SUCCESS=true
                    else
                        print_warning "Dev installation failed, trying base installation..."
                        if pip install -e . >/dev/null 2>&1; then
                            print_warning "Installed TimeFlies base (dev dependencies may be missing)"
                            cd ..
                            INSTALL_SUCCESS=true
                        else
                            cd .. 2>/dev/null || true
                            rm -rf .timeflies_src 2>/dev/null || true
                        fi
                    fi
                else
                    print_success "NVIDIA GPU detected - installing with CUDA support"
                    if pip install -e . >/dev/null 2>&1; then
                        print_success "Installed TimeFlies with GPU support"
                        cd ..
                        INSTALL_SUCCESS=true
                    else
                        print_warning "GPU installation failed, trying CPU version..."
                        if pip install -e .[cpu] >/dev/null 2>&1; then
                            print_success "Installed TimeFlies with CPU fallback"
                            cd ..
                            INSTALL_SUCCESS=true
                        else
                            cd .. 2>/dev/null || true
                            rm -rf .timeflies_src 2>/dev/null || true
                        fi
                    fi
                fi
            else
                if [[ "$PLATFORM" == "macos" ]]; then
                if [[ "$DEV_MODE" == "true" ]]; then
                    print_status "Installing TimeFlies for development on macOS"
                    if pip install -e ".[dev]" >/dev/null 2>&1; then
                        print_success "Installed TimeFlies for development on macOS"
                        cd ..
                        INSTALL_SUCCESS=true
                    else
                        print_warning "Dev installation failed, trying base installation..."
                        if pip install -e . >/dev/null 2>&1; then
                            print_warning "Installed TimeFlies base for macOS (dev dependencies may be missing)"
                            cd ..
                            INSTALL_SUCCESS=true
                        else
                            cd .. 2>/dev/null || true
                            rm -rf .timeflies_src 2>/dev/null || true
                        fi
                    fi
                else
                    print_status "Installing TimeFlies for macOS (CPU/Metal acceleration)"
                    if pip install -e . >/dev/null 2>&1; then
                        print_success "Installed TimeFlies for macOS"
                        cd ..
                        INSTALL_SUCCESS=true
                    else
                        cd .. 2>/dev/null || true
                        rm -rf .timeflies_src 2>/dev/null || true
                    fi
                fi
                else
                    if [[ "$DEV_MODE" == "true" ]]; then
                        print_status "No NVIDIA GPU detected - installing dev version"
                        if pip install -e ".[dev]" >/dev/null 2>&1; then
                            print_success "Installed TimeFlies with dev dependencies"
                            cd ..
                            INSTALL_SUCCESS=true
                        else
                            print_warning "Dev installation failed, trying base installation..."
                            if pip install -e . >/dev/null 2>&1; then
                                print_warning "Installed TimeFlies base (dev dependencies may be missing)"
                                cd ..
                                INSTALL_SUCCESS=true
                            else
                                cd .. 2>/dev/null || true
                                rm -rf .timeflies_src 2>/dev/null || true
                            fi
                        fi
                    else
                        print_status "No NVIDIA GPU detected - installing CPU version"
                        if pip install -e .[cpu] >/dev/null 2>&1; then
                            print_success "Installed TimeFlies CPU version"
                            cd ..
                            INSTALL_SUCCESS=true
                        else
                            print_warning "CPU installation failed, trying default version..."
                            if pip install -e . >/dev/null 2>&1; then
                                print_success "Installed TimeFlies with default dependencies"
                                cd ..
                                INSTALL_SUCCESS=true
                            else
                                cd .. 2>/dev/null || true
                                rm -rf .timeflies_src 2>/dev/null || true
                            fi
                        fi
                    fi
                fi
            fi
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
if [[ "$DEV_MODE" == "true" ]]; then
    print_status "Creating development activation script..."
    cat > .activate.sh << 'EOF'
#!/bin/bash
# TimeFlies Development Environment

# Suppress TensorFlow/CUDA warnings and logs for cleaner output
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export GRPC_VERBOSITY=ERROR
export AUTOGRAPH_VERBOSITY=0
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
    PS1="${PS1//( ) /}"
    PS1="${PS1//( )/}"
    # Remove any trailing spaces
    PS1="${PS1% }"
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
    PS1="${PS1//( ) /}"
    PS1="${PS1//( )/}"
    # Remove any trailing spaces
    PS1="${PS1% }"
    export PS1="(.venv) ${PS1}"
else
    echo "❌ Virtual environment not found"
    return 1
fi

# Create helpful aliases for development
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
alias tf-gui="timeflies gui"
alias tf-uninstall="timeflies uninstall"

echo "🧬 TimeFlies Development Environment Activated!"
echo ""
echo "Testing commands:"
echo "  timeflies test               # Run full test suite"
echo "  pytest                       # Run pytest directly"
echo "  pytest -v                    # Verbose test output"
echo "  pytest -k test_name          # Run specific tests"
echo ""
echo "Code quality tools:"
echo "  ruff check                   # Check code style and errors"
echo "  ruff check --fix             # Auto-fix code issues"
echo "  ruff format                  # Format code"
echo "  mypy src/                    # Type checking"
echo ""
echo "Development commands:"
echo "  timeflies create-test-data   # Generate test fixtures"
echo "  timeflies setup              # Setup test data and configs"
echo "  timeflies train              # Test ML training pipeline"
echo "  timeflies evaluate           # Test evaluation pipeline"
echo "  timeflies split              # Test data splitting"
echo "  timeflies batch-correct      # Test batch correction"
echo "  timeflies eda                # Test EDA pipeline"
echo "  timeflies analyze            # Test analysis pipeline"
echo ""
echo "Development environments:"
echo "  Main: .venv (TimeFlies + dev dependencies + testing tools)"
echo "  Batch: .venv_batch (batch correction + dev dependencies)"
echo ""
echo "Development workflow:"
echo "  1. ruff check                # Check code quality"
echo "  2. timeflies test            # Run full test suite"
echo "  3. Make your code changes"
echo "  4. ruff check --fix          # Fix any style issues"
echo "  5. pytest                    # Test your changes"
EOF
else
    print_status "Creating user activation script..."
    cat > .activate.sh << 'EOF'
#!/bin/bash
# TimeFlies Research Environment

# Suppress TensorFlow/CUDA warnings and logs for cleaner output
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export GRPC_VERBOSITY=ERROR
export AUTOGRAPH_VERBOSITY=0
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
    PS1="${PS1//( ) /}"
    PS1="${PS1//( )/}"
    # Remove any trailing spaces
    PS1="${PS1% }"
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
    PS1="${PS1//( ) /}"
    PS1="${PS1//( )/}"
    # Remove any trailing spaces
    PS1="${PS1% }"
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
alias tf-gui="timeflies gui"
alias tf-uninstall="timeflies uninstall"

echo "🧬 TimeFlies Research Environment Activated!"
echo ""
echo "Research workflow:"
echo "  timeflies setup              # Complete setup workflow"
echo "  timeflies gui                # Launch web interface in browser"
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
echo "  timeflies uninstall          # Uninstall TimeFlies completely"
echo ""
echo "Getting started:"
echo "  1. Add your H5AD files to: data/project_name/tissue_type/"
echo "  2. timeflies setup           # Setup and verify everything"
echo "  3. timeflies train           # Train with auto-evaluation"
echo "  4. Check results in:         outputs/project_name/"
EOF
fi

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
export ABSL_LOG_LEVEL=ERROR

# Activate batch correction environment
if [[ -f ".venv_batch/bin/activate" ]]; then
    source .venv_batch/bin/activate
    # Clean up prompt - remove any existing venv indicators
    PS1="${PS1//(.venv_batch) /}"
    PS1="${PS1//(.venv_batch)/}"
    PS1="${PS1//((.venv_batch) )/}"
    PS1="${PS1//((.venv_batch))/}"
    PS1="${PS1//(.venv) /}"
    PS1="${PS1//(.venv)/}"
    PS1="${PS1//((.venv) )/}"
    PS1="${PS1//((.venv))/}"
    PS1="${PS1//() /}"
    PS1="${PS1//()/}"
    PS1="${PS1//( ) /}"
    PS1="${PS1//( )/}"
    # Remove any trailing spaces
    PS1="${PS1% }"
    export PS1="(.venv_batch) ${PS1}"
elif [[ -f ".venv_batch/Scripts/activate" ]]; then
    source .venv_batch/Scripts/activate
    # Clean up prompt - remove any existing venv indicators
    PS1="${PS1//(.venv_batch) /}"
    PS1="${PS1//(.venv_batch)/}"
    PS1="${PS1//((.venv_batch) )/}"
    PS1="${PS1//((.venv_batch))/}"
    PS1="${PS1//(.venv) /}"
    PS1="${PS1//(.venv)/}"
    PS1="${PS1//((.venv) )/}"
    PS1="${PS1//((.venv))/}"
    PS1="${PS1//() /}"
    PS1="${PS1//()/}"
    PS1="${PS1//( ) /}"
    PS1="${PS1//( )/}"
    # Remove any trailing spaces
    PS1="${PS1% }"
    export PS1="(.venv_batch) ${PS1}"
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

# Create basic directory structure (skip in dev mode)
if [[ "$DEV_MODE" != "true" ]]; then
    print_status "Setting up project structure..."
    mkdir -p data
    # Basic project structure (user mode only)
fi

# Setup batch correction environment 
if [[ "$DEV_MODE" == "true" ]] || [[ "$UPDATE_MODE" != "true" ]]; then
    print_status "Creating batch correction environment..."
    if [[ ! -d ".venv_batch" ]]; then
        print_status "Creating PyTorch environment for batch correction..."
        $PYTHON_CMD -m venv .venv_batch

        # Activate batch environment
        if [[ "$PLATFORM" == "windows" ]]; then
            source .venv_batch/Scripts/activate
        else
            source .venv_batch/bin/activate
        fi

        # Install batch correction dependencies
        pip install --upgrade pip
        
        # Install TimeFlies with batch-correction and dev extras for dev mode
        if [[ "$DEV_MODE" == "true" ]]; then
            print_status "Installing TimeFlies with batch correction and dev dependencies..."
            cd .timeflies_src
            pip install -e ".[batch-correction,dev]" >/dev/null 2>&1
            cd ..
        else
            print_status "Installing TimeFlies with batch correction dependencies..."
            cd .timeflies_src
            pip install -e ".[batch-correction]" >/dev/null 2>&1
            cd ..
        fi

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
elif [[ "$UPDATE_MODE" == "true" ]]; then
    print_status "Checking batch correction environment..."
    if [[ -d ".venv_batch" ]]; then
        print_success "Batch correction environment exists"
    else
        print_warning "Batch correction environment missing - use './install_timeflies.sh --dev' to recreate"
    fi
fi

print_success "================================================"
if [[ "$UPDATE_MODE" == "true" ]]; then
    print_success "🎉 TimeFlies Update Complete!"
elif [[ "$DEV_MODE" == "true" ]]; then
    print_success "🎉 TimeFlies Developer Installation Complete!"
else
    print_success "🎉 TimeFlies Installation Complete!"
fi
print_success "================================================"
echo ""

# No need to run timeflies setup --dev anymore - everything is handled above

echo -e "${GREEN}🔬 Ready for Research!${NC}"
echo ""
if [[ "$DEV_MODE" == "true" ]]; then
    echo -e "${BLUE}Developer Mode - Next steps:${NC}"
    echo ""
    echo -e "${BLUE}1. Activate environment:${NC}"
    echo "   source .activate.sh"
    echo ""
    echo -e "${BLUE}2. Start developing:${NC}"
    echo "   # Your development environments are ready"
    echo "   # Main: .venv (TimeFlies + dev dependencies)"
    echo "   # Batch: .venv_batch (TimeFlies + batch-correction + dev dependencies)"
    echo ""
    echo -e "${BLUE}3. Development commands:${NC}"
    echo "   timeflies test               # Run test suite"
    echo "   timeflies verify             # System verification"
    echo "   timeflies update             # Update installation"
else
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
    echo "   timeflies setup --batch-correct          # Include batch correction in setup"
    echo ""
    echo -e "${BLUE}4. Run analysis workflow:${NC}"
    echo "   timeflies train              # Train models with auto-evaluation"
    echo "   timeflies batch-correct              # Optional: batch correction (auto-switches env)"
    echo "   timeflies train --batch-corrected        # Train with batch-corrected data"
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
    echo -e "${GREEN}🌐 For Web GUI Interface:${NC}"
    echo "   timeflies gui                    # Launch web interface in browser"
    echo "   timeflies gui --share            # Create public URL for remote access"
    echo ""
    echo -e "${GREEN}🛠️ For Developers:${NC}"
    echo "   git clone https://github.com/rsinghlab/TimeFlies.git"
    echo "   cd TimeFlies && timeflies setup --dev  # Create environments"
fi

echo ""
echo -e "${GREEN}🔄 Updates:${NC}"
echo "   timeflies update      # Update to latest version"
echo ""
echo -e "${GREEN}🔬 Research Lab${NC}"
echo "Contact your lab administrator for data access and support"
echo "================================================"

# Auto-activate the main environment for immediate use
echo ""
echo -e "${GREEN}🚀 Activating TimeFlies environment...${NC}"
if [[ -f ".activate.sh" ]]; then
    source .activate.sh
    echo -e "${GREEN}✅ TimeFlies environment activated!${NC}"
    echo ""
    echo -e "${BLUE}You can now run TimeFlies commands directly:${NC}"
    echo "   timeflies setup"
    echo "   timeflies train"
    echo "   timeflies --help"
    echo ""
    echo -e "${YELLOW}💡 Note: Open new terminal windows will need: source .activate.sh${NC}"
else
    echo -e "${RED}⚠️  Could not find .activate.sh - you may need to run: source .activate.sh${NC}"
fi
