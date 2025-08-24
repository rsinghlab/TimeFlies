"""
System verification and health checks for TimeFlies.

This module provides functionality to verify that the TimeFlies environment
is properly set up and ready for use.
"""

import importlib
import sys
from pathlib import Path


def verify_system(dev_mode: bool = None) -> bool:
    """
    Comprehensive system verification for TimeFlies.

    Args:
        dev_mode: If True, skip user-specific checks. If None, auto-detect.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    # Auto-detect dev mode if not specified
    if dev_mode is None:
        from pathlib import Path

        dev_mode = Path("tests").exists() and Path("src").exists()

    mode_text = "Development" if dev_mode else "System"
    print(f"SEARCH: TimeFlies {mode_text} Verification")
    print("=" * 50)

    all_checks_passed = True

    # Check Python version
    python_check = check_python_version()
    all_checks_passed &= python_check

    # Check required packages
    packages_check = check_required_packages()
    all_checks_passed &= packages_check

    # Check directory structure (different for dev mode)
    structure_check = check_directory_structure(dev_mode=dev_mode)
    all_checks_passed &= structure_check

    # Check configuration
    config_check = check_configuration()
    all_checks_passed &= config_check

    # Check analysis templates
    templates_check = check_analysis_templates()
    all_checks_passed &= templates_check

    # Check data availability (skip for dev mode)
    if not dev_mode:
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
    print("\nINFO: Python Version Check")
    print("-" * 30)

    current_version = sys.version_info
    required_version = (3, 12)  # Updated to match pyproject.toml

    if current_version >= required_version:
        print(
            f"✅ Python {current_version.major}.{current_version.minor}.{current_version.micro} (meets requirement ≥{required_version[0]}.{required_version[1]})"
        )
        return True
    else:
        print(
            f"❌ Python {current_version.major}.{current_version.minor}.{current_version.micro} (requires ≥{required_version[0]}.{required_version[1]})"
        )
        return False


def check_required_packages() -> bool:
    """Check if required packages are installed."""
    print("\nPACKAGE: Package Dependencies Check")
    print("-" * 30)

    # (package_name, min_version, import_name)
    required_packages = [
        ("tensorflow", "2.13.0", "tensorflow"),
        ("scikit-learn", "1.3.0", "sklearn"),
        ("pandas", "2.0.0", "pandas"),
        ("numpy", "1.24.0", "numpy"),
        ("scanpy", "1.9.0", "scanpy"),
        ("anndata", "0.9.0", "anndata"),
        ("matplotlib", "3.7.0", "matplotlib"),
        ("seaborn", "0.12.0", "seaborn"),
        ("shap", "0.42.0", "shap"),
        ("pyyaml", "6.0", "yaml"),
        ("dill", "0.3.0", "dill"),
        ("xgboost", "1.7.0", "xgboost"),
    ]

    all_installed = True

    for package_name, min_version, import_name in required_packages:
        try:
            package = importlib.import_module(import_name)
            installed_version = getattr(package, "__version__", "unknown")
            print(f"✅ {package_name}: {installed_version}")
        except ImportError:
            print(f"❌ {package_name}: not installed")
            all_installed = False

    return all_installed


def check_directory_structure(dev_mode: bool = False) -> bool:
    """Check if required directories exist."""
    print("\nFOUND: Directory Structure Check")
    print("-" * 30)

    if dev_mode:
        # For development, only check source directories
        required_dirs = [
            "src/common",
            "configs",
            "tests",
        ]
    else:
        # For users, check user-facing directories
        required_dirs = [
            "configs",
            "data",
            "outputs",
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

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required top-level keys
        required_keys = ["project", "data"]
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
    print("\nDATA: Data Availability Check")
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
    splits_found = False

    for project_dir in project_dirs:
        tissue_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        for tissue_dir in tissue_dirs:
            h5ad_files = list(tissue_dir.glob("*.h5ad"))
            if h5ad_files:
                print(
                    f"✅ Found {len(h5ad_files)} H5AD files in {project_dir.name}/{tissue_dir.name}"
                )
                data_found = True

                # Check if data has been properly split
                original_files = list(tissue_dir.glob("*_original.h5ad"))
                train_files = list(tissue_dir.glob("*_train.h5ad"))
                eval_files = list(tissue_dir.glob("*_eval.h5ad"))

                if original_files and train_files and eval_files:
                    print(
                        f"   ✅ Data splits ready (original: {len(original_files)}, train: {len(train_files)}, eval: {len(eval_files)})"
                    )
                    splits_found = True
                elif original_files:
                    print(
                        "   ⚠️  Data not split yet - run 'timeflies setup' or 'timeflies split'"
                    )
                else:
                    print(
                        "   ⚠️  No *_original.h5ad files found - add your source data first"
                    )

    if not data_found:
        print("⚠️  No H5AD data files found")
        print(
            "   Add your *_original.h5ad files to data/[project]/[tissue]/ directories"
        )
    elif not splits_found:
        print("NOTE: Next step: Run 'timeflies setup' to split your data for training")

    return True  # Data is not required for basic system verification


def check_gpu_availability() -> bool:
    """Check GPU availability (optional)."""
    print("\nSYSTEM:️  GPU Check (Optional)")
    print("-" * 30)

    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
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


def check_analysis_templates() -> bool:
    """Check if analysis templates directory exists and list available templates."""
    print("\nRESEARCH: Analysis Templates Check")
    print("-" * 30)

    templates_dir = Path("templates")

    if not templates_dir.exists():
        print("⚠️  templates/ directory not found")
        print("   Analysis templates are optional but recommended for custom analysis")
        return True  # Templates are optional

    print("✅ templates/ directory exists")

    # Find analysis template files
    analysis_templates = list(templates_dir.glob("*_analysis.py"))

    if not analysis_templates:
        print("ℹ️  No project analysis templates found")
        print("   Create templates/{project}_analysis.py for custom analysis")
    else:
        print(f"✅ Found {len(analysis_templates)} analysis templates:")
        for template in sorted(analysis_templates):
            template_name = template.stem.replace("_analysis", "")
            print(f"   DOC: {template.name} -> project '{template_name}'")

    # Find other template files
    other_templates = [
        f for f in templates_dir.glob("*.py") if not f.name.endswith("_analysis.py")
    ]
    list(templates_dir.glob("*example*.py"))
    readme_files = list(templates_dir.glob("README*"))

    if other_templates or readme_files:
        print("ℹ️  Additional template files:")
        for template in sorted(other_templates + readme_files):
            if template.name.startswith("README"):
                print(f"   INFO: {template.name} (documentation)")
            else:
                print(f"   DOC: {template.name}")

    return True


if __name__ == "__main__":
    verify_system()
