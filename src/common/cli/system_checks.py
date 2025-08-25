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

    # Check timeflies launcher installation
    launcher_check = check_timeflies_launcher()
    all_checks_passed &= launcher_check

    # Check model queue configurations
    queue_check = check_model_queue_configs()
    all_checks_passed &= queue_check

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
    required_version = (3, 12)  # Updated to match actual requirements

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

    all_valid = True
    configs_dir = Path("configs")

    # Check configs directory exists
    if not configs_dir.exists():
        print("❌ configs/ directory not found")
        return False

    # Check each required config file
    required_configs = {
        "default.yaml": ["project", "data", "model"],
        "setup.yaml": ["train_test_split", "sampling"],
        "batch_correction.yaml": ["batch_correction", "pytorch"],
        "hyperparameter_tuning.yaml": ["method", "model_hyperparams"],
        "model_queue.yaml": None,  # No specific required keys
    }

    import yaml

    for config_file, required_keys in required_configs.items():
        config_path = configs_dir / config_file

        if not config_path.exists():
            if config_file in ["default.yaml", "setup.yaml"]:
                print(f"❌ {config_file} not found (required)")
                all_valid = False
            else:
                print(f"ℹ️  {config_file} not found (optional)")
            continue

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required keys if specified
            if required_keys:
                missing_keys = [key for key in required_keys if key not in config]
                if missing_keys:
                    print(f"❌ {config_file} missing keys: {missing_keys}")
                    all_valid = False
                else:
                    print(f"✅ {config_file} valid")
            else:
                print(f"✅ {config_file} found")

        except Exception as e:
            print(f"❌ {config_file} error: {e}")
            all_valid = False

    return all_valid


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

                # Check for batch-corrected files
                batch_original_files = list(tissue_dir.glob("*_original_batch.h5ad"))
                batch_train_files = list(tissue_dir.glob("*_train_batch.h5ad"))
                batch_eval_files = list(tissue_dir.glob("*_eval_batch.h5ad"))

                if original_files and train_files and eval_files:
                    print(
                        f"   ✅ Data splits ready (original: {len(original_files)}, train: {len(train_files)}, eval: {len(eval_files)})"
                    )
                    splits_found = True

                    # Show batch-corrected files if they exist
                    if batch_original_files and batch_train_files and batch_eval_files:
                        print(
                            f"   ✅ Batch-corrected splits available (original: {len(batch_original_files)}, train: {len(batch_train_files)}, eval: {len(batch_eval_files)})"
                        )
                elif original_files:
                    print(
                        "   ⚠️  Data not split yet - run 'timeflies setup' or 'timeflies split'"
                    )
                else:
                    print(
                        "   ⚠️  No *_original.h5ad files found - add your source data first"
                    )

                # Check for unexpected files
                expected_patterns = [
                    "*_original.h5ad",
                    "*_train.h5ad",
                    "*_eval.h5ad",
                    "*_original_batch.h5ad",
                    "*_train_batch.h5ad",
                    "*_eval_batch.h5ad",
                    "*.csv",
                ]  # Allow CSV files like autosomal.csv, sex.csv
                expected_files = set()
                for pattern in expected_patterns:
                    expected_files.update([f.name for f in tissue_dir.glob(pattern)])

                all_files = set([f.name for f in tissue_dir.glob("*") if f.is_file()])
                unexpected_files = all_files - expected_files

                if unexpected_files:
                    print(
                        f"   ⚠️  Additional files found: {', '.join(sorted(unexpected_files))}"
                    )
                    print("      Please verify these files are needed")

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


def check_timeflies_launcher() -> bool:
    """Check if timeflies launcher is properly installed and accessible."""
    print("\nLAUNCHER: TimeFlies CLI Installation Check")
    print("-" * 30)

    import shutil
    import subprocess

    # Check if timeflies command is available in PATH
    if shutil.which("timeflies"):
        print("[OK] timeflies command found in PATH")

        # Test if command works
        try:
            result = subprocess.run(
                ["timeflies", "--help"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("[OK] timeflies command working correctly")
                return True
            else:
                print(f"[ERROR] timeflies command failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Could not test timeflies command: {e}")
            return False
    else:
        print("[ERROR] timeflies command not found in PATH")
        print("   Run: pip install -e . (from TimeFlies directory)")
        print("   Or: source .activate.sh (if using installed version)")
        return False


def check_model_queue_configs() -> bool:
    """Check if model queue configurations are available and accessible."""
    print("\nQUEUE: Model Queue Configuration Check")
    print("-" * 30)

    configs_dir = Path("configs")

    if not configs_dir.exists():
        print("[ERROR] configs/ directory not found")
        print("   Model queue configurations require configs/ directory")
        return False

    print("[OK] configs/ directory exists")

    # Check for default model queue configuration
    default_queue_config = configs_dir / "model_queue.yaml"

    if default_queue_config.exists():
        print("[OK] Default model queue configuration found: model_queue.yaml")

        # Validate the default config
        try:
            import yaml

            with open(default_queue_config) as f:
                queue_data = yaml.safe_load(f)
            if "model_queue" in queue_data and "global_settings" in queue_data:
                model_count = len(queue_data.get("model_queue", []))
                print(f"   └─ {model_count} models configured in default queue")
            else:
                print("   └─ [WARN] Default config missing required sections")
        except Exception as e:
            print(f"   └─ [ERROR] Default config invalid: {e}")
    else:
        print("[WARN] No default model queue configuration found")
        print("   Create configs/model_queue.yaml for 'timeflies queue' command")

    # Look for additional model queue configuration files
    queue_configs = [
        f for f in configs_dir.glob("model_queue*.yaml") if f.name != "model_queue.yaml"
    ]

    if queue_configs:
        print(f"[OK] Found {len(queue_configs)} additional queue configurations:")
        for config in sorted(queue_configs):
            print(f"   CONFIG: {config.name}")

            # Validate YAML structure
            try:
                import yaml

                with open(config) as f:
                    queue_data = yaml.safe_load(f)

                if "model_queue" in queue_data and "global_settings" in queue_data:
                    model_count = len(queue_data.get("model_queue", []))
                    print(f"      └─ {model_count} models configured")
                else:
                    print("      └─ [WARN] Missing required sections")

            except Exception as e:
                print(f"      └─ [ERROR] Invalid YAML: {e}")

    # Check if user has access to create queue configs in current directory
    try:
        test_file = Path("configs/test_queue_write.tmp")
        configs_dir.mkdir(exist_ok=True)
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        print("[OK] User can create queue configurations")
    except Exception as e:
        print(f"[WARN] Cannot write to configs/: {e}")
        print("   User may need write permissions for model queue configs")

    return True


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
