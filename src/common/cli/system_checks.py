"""
System verification and health checks for TimeFlies.

This module provides functionality to verify that the TimeFlies environment
is properly set up and ready for use.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple
import pkg_resources


def verify_system() -> bool:
    """
    Comprehensive system verification for TimeFlies.
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print("🔍 TimeFlies System Verification")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check Python version
    python_check = check_python_version()
    all_checks_passed &= python_check
    
    # Check required packages
    packages_check = check_required_packages()
    all_checks_passed &= packages_check
    
    # Check directory structure
    structure_check = check_directory_structure()
    all_checks_passed &= structure_check
    
    # Check configuration
    config_check = check_configuration()
    all_checks_passed &= config_check
    
    # Check data availability
    data_check = check_data_availability()
    all_checks_passed &= data_check
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ All system checks passed! TimeFlies is ready to use.")
        return True
    else:
        print("❌ Some checks failed. Please address the issues above.")
        return False


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    print("\n📋 Python Version Check")
    print("-" * 30)
    
    current_version = sys.version_info
    required_version = (3, 12)  # Updated to match pyproject.toml
    
    if current_version >= required_version:
        print(f"✅ Python {current_version.major}.{current_version.minor}.{current_version.micro} (meets requirement ≥{required_version[0]}.{required_version[1]})")
        return True
    else:
        print(f"❌ Python {current_version.major}.{current_version.minor}.{current_version.micro} (requires ≥{required_version[0]}.{required_version[1]})")
        return False


def check_required_packages() -> bool:
    """Check if required packages are installed."""
    print("\n📦 Package Dependencies Check")
    print("-" * 30)
    
    required_packages = [
        ("tensorflow", "2.13.0"),
        ("scikit-learn", "1.3.0"),
        ("pandas", "2.0.0"),
        ("numpy", "1.24.0"),
        ("scanpy", "1.9.0"),
        ("anndata", "0.9.0"),
        ("matplotlib", "3.7.0"),
        ("seaborn", "0.12.0"),
        ("shap", "0.42.0"),
        ("pyyaml", "6.0"),
        ("dill", "0.3.0"),
        ("xgboost", "1.7.0"),
    ]
    
    all_installed = True
    
    for package_name, min_version in required_packages:
        try:
            package = importlib.import_module(package_name.replace("-", "_"))
            installed_version = getattr(package, '__version__', 'unknown')
            print(f"✅ {package_name}: {installed_version}")
        except ImportError:
            print(f"❌ {package_name}: not installed")
            all_installed = False
    
    return all_installed


def check_directory_structure() -> bool:
    """Check if required directories exist."""
    print("\n📁 Directory Structure Check")
    print("-" * 30)
    
    required_dirs = [
        "src/common",
        "src/analysis", 
        "configs",
        "data",
        "docs",
    ]
    
    all_exist = True
    current_dir = Path.cwd()
    
    for dir_path in required_dirs:
        full_path = current_dir / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_configuration() -> bool:
    """Check if configuration files exist and are valid."""
    print("\n⚙️  Configuration Check")
    print("-" * 30)
    
    config_path = Path("configs/default.yaml")
    
    if not config_path.exists():
        print("❌ configs/default.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required top-level keys
        required_keys = ['project', 'data']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        
        print(f"✅ Configuration valid (project: {config.get('project', 'unknown')})")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def check_data_availability() -> bool:
    """Check if data directories and files are set up."""
    print("\n📊 Data Availability Check")
    print("-" * 30)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("⚠️  No data directory found")
        print("   Create data/ and add your H5AD files to get started")
        return True  # Not required for initial setup
    
    # Look for project directories
    project_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not project_dirs:
        print("⚠️  No project directories found in data/")
        print("   Add your data to data/[project]/[tissue]/ to get started")
        return True  # Not required for initial setup
    
    # Check for data files in each project
    data_found = False
    for project_dir in project_dirs:
        tissue_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        for tissue_dir in tissue_dirs:
            h5ad_files = list(tissue_dir.glob("*.h5ad"))
            if h5ad_files:
                print(f"✅ Found {len(h5ad_files)} H5AD files in {project_dir.name}/{tissue_dir.name}")
                data_found = True
    
    if not data_found:
        print("⚠️  No H5AD data files found")
        print("   Add your *_original.h5ad files to data/[project]/[tissue]/ directories")
    
    return True  # Data is not required for basic system verification


def check_gpu_availability() -> bool:
    """Check GPU availability (optional)."""
    print("\n🖥️  GPU Check (Optional)")
    print("-" * 30)
    
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("ℹ️  No GPUs detected - CPU mode will be used")
            return True
    except Exception as e:
        print(f"⚠️  Could not check GPU status: {e}")
        return True  # GPU is optional


if __name__ == "__main__":
    verify_system()