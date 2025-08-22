#!/usr/bin/env python3
"""
TimeFlies - Machine Learning for Aging Analysis

A modern machine learning framework for analyzing aging patterns in 
Drosophila single-cell RNA sequencing data.

Main Commands:
  verify            - Verify environment and setup
  create-test-data  - Create test fixtures from real data
  setup             - Create train/eval data splits
  train             - Train models (auto-detects active project)
  evaluate          - Evaluate pre-trained models
  test              - Run test suite

Workflow:
  1. Place data in data/[project]/[tissue]/
  2. python run_timeflies.py create-test-data
  3. python run_timeflies.py setup 
  4. python run_timeflies.py verify 
  5. python run_timeflies.py train
  6. python run_timeflies.py evaluate (optional)

Switch Projects:
  Edit configs/default.yaml and change 'project: fruitfly_aging' to 'project: fruitfly_alzheimers'

For detailed help:
  python run_timeflies.py --help
  python run_timeflies.py COMMAND --help

Author: Singh Lab
Repository: https://github.com/rsinghlab/TimeFlies
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow and CUDA warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops message
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# Suppress ABSL warnings
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

def show_banner():
    """Show TimeFlies banner and basic info."""
    print("=" * 60)
    print("TIMEFLIES - Machine Learning for Aging Analysis")
    print("=" * 60)
    print("Single-cell RNA-seq analysis for Drosophila aging patterns")
    print("Deep learning models with SHAP interpretability")
    print("Comprehensive visualization and evaluation tools")
    print("=" * 60)

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
        print("Troubleshooting:")
        print("   1. Make sure you're in the TimeFlies root directory")
        print("   2. Activate your virtual environment: source activate.sh")
        print("   3. Install dependencies: bash setup_dev_env.sh")
        print()
        print("Current directory:", current_dir)
        print("Looking for src at:", src_path)
        sys.exit(1)

    # Show banner for help or no arguments
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        show_banner()
        print()
        
    # Run the CLI and pipeline
    try:
        # Only show "Starting" message for specific setup commands
        if len(sys.argv) > 1 and sys.argv[1] in ['setup', 'create-test-data']:
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