#!/usr/bin/env python3
"""
TimeFlies - Streamlined Entry Point

A modern machine learning framework for analyzing aging patterns in 
Drosophila single-cell RNA sequencing data.

Usage:
    python run_timeflies.py [options]
    
For detailed help:
    python run_timeflies.py --help

Author: Singh Lab
Repository: https://github.com/rsinghlab/TimeFlies
"""

import sys
from pathlib import Path

# Add src to Python path for imports
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from timeflies.cli import main_cli
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the TimeFlies root directory.")
    sys.exit(1)


if __name__ == "__main__":
    exit_code = main_cli()
    sys.exit(exit_code)