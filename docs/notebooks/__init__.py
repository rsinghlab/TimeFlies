"""
TimeFlies: Machine Learning for Aging Analysis

A modern, notebook-friendly machine learning framework for analyzing aging patterns
in Drosophila single-cell RNA sequencing data.

Quick Start (Notebook):
    >>> import timeflies
    >>> timeflies.setup_environment()  # One-time setup
    >>> config = timeflies.Config(project="alzheimers", tissue="head", model="cnn")
    >>> results = timeflies.train(config)
    >>> analysis = timeflies.analyze(results)
    >>> eda = timeflies.eda(config, save_report=True)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# Version info
__version__ = "0.3.0"
__author__ = "Singh Lab"

# Easy imports for notebook use
from .notebook import Config, train, analyze, eda, evaluate, get_experiment_results


def setup_environment(skip_batch: bool = False, data_check: bool = True) -> bool:
    """
    🚀 One-command setup for TimeFlies in notebooks.
    
    This runs the complete setup process:
    - Verifies system requirements
    - Creates test data
    - Sets up data splits  
    - Creates output directories
    - Optionally sets up batch correction
    
    Args:
        skip_batch: Skip batch correction setup (faster)
        data_check: Check if data files exist before setup
        
    Returns:
        bool: True if setup successful
        
    Example:
        >>> import timeflies
        >>> timeflies.setup_environment()
        🚀 TimeFlies Unified Setup
        ================================
        ✅ Setup complete! Ready to train models.
    """
    try:
        # Get the project root
        current_file = Path(__file__).parent.parent.parent
        project_root = current_file
        
        # Check if we're in the right directory
        if not (project_root / "run_timeflies.py").exists():
            print("❌ TimeFlies not found. Please ensure you're in the TimeFlies directory.")
            return False
            
        # Check for data if requested
        if data_check:
            data_dir = project_root / "data"
            if not data_dir.exists() or not list(data_dir.glob("**/*.h5ad")):
                print("⚠️  No data files found in data/ directory.")
                print("   Please place your H5AD files in data/[project]/[tissue]/")
                print("   Example: data/fruitfly_alzheimers/head/your_file.h5ad")
                return False
        
        # Build command
        cmd = [
            sys.executable, 
            str(project_root / "run_timeflies.py"),
            "setup-all"
        ]
        
        if skip_batch:
            cmd.append("--skip-batch")
            
        # Set environment
        env = {"PYTHONPATH": str(project_root / "src")}
        
        # Run setup
        result = subprocess.run(
            cmd, 
            cwd=project_root,
            env={**os.environ, **env},
            capture_output=False
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def info():
    """Display TimeFlies information and quick help."""
    print(f"""
🔬 TimeFlies v{__version__} - Machine Learning for Aging Analysis
================================================================

Quick Start:
  1. Place H5AD files in data/[project]/[tissue]/
  2. Run: timeflies.setup_environment()  
  3. Configure and train: 
     config = timeflies.Config(project="alzheimers")
     results = timeflies.train(config)

Available Projects:
  • fruitfly_aging - Healthy aging patterns
  • fruitfly_alzheimers - Disease model analysis
  
Available Models:
  • CNN - Convolutional neural network (default)
  • MLP - Multi-layer perceptron  
  • XGBoost - Gradient boosting
  • Random Forest - Tree ensemble
  • Logistic - Linear classification

Documentation: https://github.com/rsinghlab/TimeFlies
""")


def version():
    """Get TimeFlies version."""
    return __version__


# Make key functions available at package level
__all__ = [
    # Setup
    "setup_environment",
    "info", 
    "version",
    
    # Core functionality
    "Config",
    "train",
    "analyze", 
    "eda",
    "evaluate",
    "get_experiment_results",
]