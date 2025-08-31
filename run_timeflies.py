#!/usr/bin/env python3
"""
TimeFlies v1.0 - Machine Learning for Aging Analysis

A comprehensive machine learning framework for analyzing aging patterns in
Drosophila single-cell RNA sequencing data with deep learning models,
SHAP interpretability, and batch correction.

Core Commands:
  setup             - Complete setup: split data + verify + create directories
  train             - Train models with automatic evaluation
  eda               - Exploratory data analysis on full dataset
  analyze           - Project-specific analysis with SHAP
  batch-correct     - Apply batch correction to data splits

Individual Commands:
  split             - Create train/eval data splits only
  verify            - Verify installation and system
  evaluate          - Evaluate trained models
  test              - Run comprehensive test suite

Data Management:
  create-test-data  - Generate 3-tier test fixtures (tiny/synthetic/real)
  update            - Update TimeFlies to latest version from GitHub

User Workflow:
  1. Place your *_original.h5ad files in data/[project]/[tissue]/
  2. Configure: nano configs/default.yaml
  3. Setup: python run_timeflies.py setup [--batch-correct]
  4. Train: python run_timeflies.py train [--with-eda --with-analysis]
  5. Evaluate: python run_timeflies.py evaluate [--with-eda --with-analysis]
  6. Analyze: python run_timeflies.py analyze [--with-eda]

Project Switching:
  --aging / --alzheimers    (temporary override)
  OR edit configs/default.yaml permanently

Global Options:
  --tissue head/body        - Override tissue type
  --batch-corrected         - Use batch-corrected data
  --verbose                 - Enable verbose logging

For detailed help:
  python run_timeflies.py --help
  python run_timeflies.py COMMAND --help

TimeFlies v1.0 | Singh Lab | https://github.com/rsinghlab/TimeFlies
"""

import os
import sys
import warnings
import subprocess
from pathlib import Path

# Python version check and auto-upgrade
def ensure_python_312():
    """Ensure we're running with Python 3.12+, re-exec if needed."""
    current_version = sys.version_info
    
    # If already running Python 3.12+, continue
    if current_version >= (3, 12):
        return
    
    # Find Python 3.12+
    for cmd in ["python3.12", "python3", "python"]:
        try:
            result = subprocess.run(
                [cmd, "--version"], capture_output=True, text=True, check=True
            )
            version_str = result.stdout.strip().split()[-1]
            major, minor = map(int, version_str.split(".")[:2])
            
            if major == 3 and minor >= 12:
                print(f"TimeFlies requires Python 3.12+, found {current_version.major}.{current_version.minor}")
                print(f"Re-executing with {cmd} {version_str}...")
                # Get full path to the Python executable
                python_path = subprocess.run(
                    ["which", cmd], capture_output=True, text=True, check=True
                ).stdout.strip()
                # Re-execute with the correct Python version
                os.execv(python_path, [python_path] + sys.argv)
                
        except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
            continue
    
    # If no Python 3.12+ found
    print("ERROR: TimeFlies requires Python 3.12+ but none found.")
    print("Please install Python 3.12+ or run:")
    print("  brew install python@3.12  (on macOS)")
    print("  sudo apt install python3.12  (on Ubuntu/Debian)")
    sys.exit(1)

# Check Python version first thing
ensure_python_312()


def handle_dev_setup():
    """Handle dev setup directly without importing CLI modules."""
    print("SETUP: TimeFlies Developer Setup")
    print("=" * 50)
    print("Setting up development environments...")
    
    try:
        # Find Python 3.12+ command
        python_cmd = None
        for cmd in ["python3.12", "python3", "python"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"], capture_output=True, text=True, check=True
                )
                version = result.stdout.strip().split()[-1]
                major, minor = map(int, version.split(".")[:2])
                if major == 3 and minor >= 12:
                    python_cmd = cmd
                    print(f"[OK] Found Python {version}")
                    break
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                continue

        if not python_cmd:
            print("[ERROR] Python 3.12+ required but not found")
            return 1

        # Create main environment
        print("\nPYTHON: Setting up main environment (.venv)...")
        if Path(".venv").exists():
            print("SKIP: Removing existing main environment...")
            subprocess.run(["rm", "-rf", ".venv"], check=True)

        subprocess.run([python_cmd, "-m", "venv", ".venv"], check=True)
        print("[OK] Main environment created")

        # Install main dependencies
        print("PACKAGE: Installing main dependencies...")
        venv_pip = ".venv/bin/pip"

        subprocess.run([venv_pip, "install", "--upgrade", "pip"], check=True)
        subprocess.run([venv_pip, "install", "-e", "."], check=True)

        # Install development dependencies
        print("SETUP: Installing development dependencies...")
        subprocess.run([venv_pip, "install", "-e", ".[dev]"], check=True)
        print("[OK] Main and development dependencies installed")

        # Create batch environment
        print("\nRESEARCH: Setting up batch correction environment (.venv_batch)...")
        if Path(".venv_batch").exists():
            print("SKIP: Removing existing batch environment...")
            subprocess.run(["rm", "-rf", ".venv_batch"], check=True)

        subprocess.run([python_cmd, "-m", "venv", ".venv_batch"], check=True)
        print("[OK] Batch environment created")

        # Install batch dependencies
        print("TESTING: Installing batch correction dependencies...")
        batch_pip = ".venv_batch/bin/pip"

        subprocess.run([batch_pip, "install", "--upgrade", "pip"], check=True)
        # Install TimeFlies with batch-correction and dev extras
        subprocess.run(
            [batch_pip, "install", "-e", ".[batch-correction,dev]"], check=True
        )
        print("[OK] Batch dependencies installed")

        # Create activation scripts
        print("\nCREATE: Creating activation scripts...")
        create_activation_scripts()
        print("[OK] Activation scripts created")

        print("\n✅ DEVELOPMENT SETUP COMPLETE!")
        print("=" * 60)
        print("\nTo activate the development environment:")
        print("  source .activate.sh")
        print(
            "\nNote: Batch correction tests will auto-switch to .venv_batch when needed."
        )
        print("To manually switch to batch environment:")
        print("  source .activate_batch.sh")

        return 0

    except Exception as e:
        print(f"[ERROR] Development setup failed: {e}")
        return 1


def create_activation_scripts():
    """Create activation scripts for both environments."""
    # Main activation script
    main_script = """#!/bin/bash
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
"""

    # Batch activation script
    batch_script = """#!/bin/bash
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
"""

    # Write the scripts
    with open(".activate.sh", "w") as f:
        f.write(main_script)
    
    with open(".activate_batch.sh", "w") as f:
        f.write(batch_script)
    
    # Make them executable
    os.chmod(".activate.sh", 0o755)
    os.chmod(".activate_batch.sh", 0o755)

# Suppress TensorFlow and CUDA warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only show errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom ops message
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_DISABLE_MKL"] = "1"  # Disable Intel MKL warnings
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Suppress ABSL and other warnings
import logging

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Additional suppression for CUDA/cuDNN warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def show_banner():
    """Show TimeFlies v1.0 banner and basic info."""
    print("=" * 70)
    print("    TIMEFLIES v1.0 - Machine Learning for Aging Analysis")
    print("=" * 70)
    print("Single-cell RNA-seq analysis for Drosophila aging patterns")
    print("Deep learning models with SHAP interpretability")
    print("Batch correction and comprehensive evaluation tools")
    print("3-tier test data system for reliable development")
    print("=" * 70)


def main():
    """Enhanced main function with better user experience."""
    # Check if this is a dev setup command that needs to create environments first
    if len(sys.argv) >= 3 and sys.argv[1] == "setup" and "--dev" in sys.argv:
        # Handle dev setup directly without importing CLI modules
        exit_code = handle_dev_setup()
        sys.exit(exit_code)
    
    # Add src to Python path for imports
    current_dir = Path(__file__).parent.absolute()
    src_path = current_dir / "src"
    sys.path.insert(0, str(src_path))

    try:
        from common.cli import main_cli
    except ImportError as e:
        print("Import Error:")
        print(f"   {e}")
        print()
        print("Current directory:", current_dir)
        print("Looking for src at:", src_path)
        print()
        print("If you're setting up for development, try:")
        print("   python3.12 run_timeflies.py setup --dev")
        sys.exit(1)

    # Show banner for help or no arguments
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        show_banner()
        print()

    # Run the CLI and pipeline
    try:
        # Only show "Starting" message for specific setup commands
        if len(sys.argv) > 1 and sys.argv[1] in ["setup", "create-test-data"]:
            print("Starting TimeFlies pipeline...")
        exit_code = main_cli()

        # Don't print additional success/failure messages - commands handle this themselves
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline Error: {e}")
        print("\nFor help: python run_timeflies.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
