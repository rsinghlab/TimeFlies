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
import subprocess
import sys
import warnings
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
                print(
                    f"TimeFlies requires Python 3.12+, found {current_version.major}.{current_version.minor}"
                )
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
        print("   ./install_timeflies.sh --dev")
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
