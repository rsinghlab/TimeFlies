"""
TimeFlies CLI Commands

This module contains all command-line interface commands for TimeFlies.
Each command is implemented as a separate function with proper error handling.
"""

import os
import subprocess
import sys
from pathlib import Path

from ..analysis.eda import EDAHandler
from ..core.active_config import get_config_for_active_project

# Optional import for batch correction
try:
    from ..data.preprocessing.batch_correction import BatchCorrector

    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    BatchCorrector = None


def run_system_tests(args) -> int:
    """Run test suite or provide helpful guidance for users."""
    from pathlib import Path

    # Check if tests directory exists (developer installation)
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("TimeFlies Test Information")
        print("=" * 50)
        print("WARNING: Full test suite not available in user installation")
        print("")
        print("To verify your installation works:")
        print("   timeflies verify")
        print("")
        print("For full development testing:")
        print("   git clone https://github.com/rsinghlab/TimeFlies.git")
        print("   cd TimeFlies")
        print("   timeflies test --coverage")
        print("")
        print("NOTE: User installation is working correctly if:")
        print("   • timeflies verify passes")
        print("   • timeflies setup completes successfully")
        print("   • Your research workflow runs without errors")
        return 0

    # Developer path - run actual tests
    print("TESTING: Running TimeFlies Test Suite...")
    print("=" * 50)

    import sys
    from pathlib import Path

    # Build test command
    cmd = [sys.executable, "-m", "pytest", "tests/"]

    # Add test type filter
    test_type = getattr(args, "test_type", "all")
    if test_type != "all":
        cmd.extend(["-m", test_type])

    # Add options
    if getattr(args, "coverage", False):
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if getattr(args, "verbose", False):
        cmd.append("-v")

    if getattr(args, "fast", False):
        cmd.extend(["-m", "not functional and not system"])

    if getattr(args, "debug", False):
        cmd.extend(["-x", "-v", "-s"])

    if getattr(args, "rerun", False):
        cmd.append("--lf")  # last failed

    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return 1


def execute_command(args) -> bool:
    """Execute the appropriate command based on parsed arguments."""
    try:
        if args.command == "train":
            config = get_config_for_active_project()
            return train_command(args, config) == 0
        elif args.command == "evaluate":
            config = get_config_for_active_project()
            return evaluate_command(args, config) == 0
        elif args.command == "eda":
            config = get_config_for_active_project()
            return eda_command(args, config) == 0
        elif args.command == "analyze":
            config = get_config_for_active_project()
            return analyze_command(args, config) == 0
        elif args.command == "batch-correct":
            return batch_command(args) == 0
        elif args.command == "setup":
            return new_setup_command(args) == 0
        elif args.command == "split":
            return split_command(args) == 0
        elif args.command == "verify":
            from common.cli.system_checks import verify_system

            # Let verify_system auto-detect dev mode unless explicitly overridden
            dev_mode = (
                args.dev if hasattr(args, "dev") and args.dev is not None else None
            )
            return verify_system(dev_mode=dev_mode) == 0
        elif args.command == "test":
            return run_system_tests(args) == 0
        elif args.command == "create-test-data":
            return create_test_data_command(args) == 0
        elif args.command == "update":
            return update_command(args) == 0
        elif args.command == "tune":
            return tune_command(args) == 0
        elif args.command == "queue":
            return queue_command(args) == 0
        else:
            print(f"Unknown command: {args.command}")
            return False
    except Exception as e:
        print(f"Command execution failed: {e}")
        return False


def verify_workflow_command(args) -> int:
    """Verify the complete workflow setup and data integrity."""
    import logging
    import os
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

    # Temporarily reduce logging level and suppress warnings
    original_level = logging.getLogger().level
    active_config_level = logging.getLogger("timeflies.common.core.active_config").level
    config_manager_level = logging.getLogger(
        "timeflies.projects.fruitfly_aging.core.config_manager"
    ).level

    logging.getLogger().setLevel(logging.ERROR)  # Only show errors
    logging.getLogger("timeflies.common.core.active_config").setLevel(logging.ERROR)
    logging.getLogger("timeflies.projects.fruitfly_aging.core.config_manager").setLevel(
        logging.ERROR
    )
    warnings.filterwarnings("ignore")  # Suppress warnings like scVI

    all_passed = True
    workflow_complete = True

    # 1. Test Python environment and imports
    print("1. Testing Python environment...")
    try:
        import anndata
        import numpy
        import pandas
        import scanpy

        print("   [OK] Scientific packages (numpy, pandas, scanpy, anndata)")

        import sklearn
        import tensorflow

        print("   [OK] Machine learning packages (tensorflow, sklearn)")

        from ..core.active_config import (
            get_active_project,
            get_config_for_active_project,
        )
        from ..utils.gpu_handler import GPUHandler

        print("   [OK] TimeFlies core modules")

    except ImportError as e:
        print(f"   [ERROR] Import failed: {e}")
        all_passed = False

    # 2. Test project configuration
    print("\n2. Testing project configuration...")
    try:
        # Check for project override from CLI flags
        if hasattr(args, "project") and args.project:
            active_project = args.project
            print(f"   PROCESS: Using CLI project override: {active_project}")
            # Use importlib to dynamically import the project's config manager
            import importlib

            config_module = importlib.import_module(
                f"projects.{args.project}.core.config_manager"
            )
            ConfigManager = config_module.ConfigManager
            config_manager = ConfigManager("default")
        else:
            active_project = get_active_project()
            print(f"   [OK] Active project detected: {active_project}")
            config_manager = get_config_for_active_project("default")

        config = config_manager.get_config()
        print("   [OK] Project config loaded successfully")
        print(f"   [OK] Target variable: {config.data.target_variable}")
        print(f"   [OK] Model type: {config.data.model}")

    except Exception as e:
        print(f"   [ERROR] Project config not set up: {e}")
        print("   NOTE: Check your configs/default.yaml file")
        all_passed = False

    # 3. Test original data files (workflow step 2)
    print("\n3. Testing original data files...")
    try:
        from pathlib import Path

        data_root = Path("data")

        if not data_root.exists():
            print("   [ERROR] No data directory found")
            print(
                "   NOTE: Create data/[project]/[tissue]/ and add original H5AD files"
            )
            workflow_complete = False
            all_passed = False
        else:
            original_files_found = []
            for project_dir in data_root.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    for tissue_dir in project_dir.iterdir():
                        if tissue_dir.is_dir():
                            original_files = list(tissue_dir.glob("*_original.h5ad"))
                            if original_files:
                                original_files_found.extend(
                                    [
                                        f"{project_dir.name}/{tissue_dir.name}: {f.name}"
                                        for f in original_files
                                    ]
                                )

            if original_files_found:
                print("   [OK] Original data files found:")
                for file in original_files_found:
                    print(f"      {file}")
            else:
                print("   [ERROR] No original H5AD data files found")
                print(
                    "   NOTE: Add your *_original.h5ad files to data/[project]/[tissue]/"
                )
                workflow_complete = False

    except Exception as e:
        print(f"   [ERROR] Data check failed: {e}")
        all_passed = False

    # 4. Test test data fixtures (workflow step 4)
    print("\n4. Testing test data fixtures...")
    try:
        from pathlib import Path

        fixtures_root = Path("tests/fixtures")

        if not fixtures_root.exists():
            print("   [ERROR] No test fixtures directory found")
            print("   NOTE: Run: python run_timeflies.py create-test-data")
            workflow_complete = False
        else:
            test_projects = []
            for project_dir in fixtures_root.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    h5ad_files = list(project_dir.glob("test_data_*.h5ad"))
                    stats_files = list(project_dir.glob("test_data_*_stats.json"))
                    if h5ad_files and stats_files:
                        test_projects.append(project_dir.name)

            if test_projects:
                print(f"   [OK] Test data available for: {', '.join(test_projects)}")

                summary_file = fixtures_root / "test_data_summary.json"
                if summary_file.exists():
                    print("   [OK] Test data summary file exists")
                else:
                    print("   WARNING:  Test data summary missing")
            else:
                print("   [ERROR] No test data fixtures found")
                print("   NOTE: Run: python run_timeflies.py create-test-data")
                workflow_complete = False

    except Exception as e:
        print(f"   [ERROR] Test data check failed: {e}")
        workflow_complete = False
        all_passed = False

    # 5. Test data splits (workflow step 5)
    print("\n5. Testing data splits...")
    try:
        from pathlib import Path

        data_root = Path("data")

        splits_found = []
        if data_root.exists():
            for project_dir in data_root.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    for tissue_dir in project_dir.iterdir():
                        if tissue_dir.is_dir():
                            train_files = list(tissue_dir.glob("*_train.h5ad"))
                            eval_files = list(tissue_dir.glob("*_eval.h5ad"))
                            if train_files and eval_files:
                                splits_found.append(
                                    f"{project_dir.name}/{tissue_dir.name}"
                                )

        if splits_found:
            print(f"   [OK] Data splits found for: {', '.join(splits_found)}")
        else:
            print("   [ERROR] No data splits (*_train.h5ad, *_eval.h5ad) found")
            print("   NOTE: Run: python run_timeflies.py setup")
            workflow_complete = False

    except Exception as e:
        print(f"   [ERROR] Data splits check failed: {e}")
        workflow_complete = False
        all_passed = False

    # 6. Test batch correction (workflow step 6, optional)
    print("\n6. Testing batch correction...")
    try:
        from pathlib import Path

        # Check if batch environment exists
        batch_env = Path(".venv_batch")
        if batch_env.exists():
            print("   [OK] Batch correction environment found")

            # Check for batch corrected files
            data_root = Path("data")
            batch_files_found = []
            if data_root.exists():
                for project_dir in data_root.iterdir():
                    if project_dir.is_dir() and not project_dir.name.startswith("."):
                        for tissue_dir in project_dir.iterdir():
                            if tissue_dir.is_dir():
                                batch_files = list(tissue_dir.glob("*_batch.h5ad"))
                                if batch_files:
                                    batch_files_found.extend(
                                        [
                                            f"{project_dir.name}/{tissue_dir.name}: {f.name}"
                                            for f in batch_files
                                        ]
                                    )

            if batch_files_found:
                print("   [OK] Batch corrected files found:")
                for file in batch_files_found:
                    print(f"      {file}")
            else:
                print(
                    "   WARNING:  Batch correction environment found but no batch corrected files"
                )
                print(
                    "   NOTE: Run: source activate_batch.sh && python run_timeflies.py batch-correct"
                )
        else:
            print("   WARNING:  No batch correction environment")
            print(
                "   NOTE: Optional: Re-run setup_dev_env.sh and choose 'y' for batch environment"
            )

    except Exception as e:
        print(f"   WARNING:  Could not check batch correction: {e}")

    # 7. Test GPU/hardware
    print("\n7. Testing hardware configuration...")
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"   [OK] GPU available: {len(gpus)} device(s)")
        else:
            print("   [OK] Running on CPU (GPU not required)")

    except Exception as e:
        print(f"   WARNING:  Could not check hardware: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_passed and workflow_complete:
        print("SUCCESS: COMPLETE WORKFLOW VERIFIED! Ready to train models.")
    elif all_passed:
        print("[OK] CORE SYSTEM OK - Complete the workflow steps below.")
    else:
        print("[ERROR] ISSUES FOUND - Fix the problems above.")

    print("\nINFO: Workflow status:")
    print(
        "   1. [OK] Data files placed"
        if workflow_complete
        else "   1. [ERROR] Place original H5AD files in data/[project]/[tissue]/"
    )
    print(
        "   2. [OK] Test fixtures created"
        if workflow_complete
        else "   2. [ERROR] Run: python run_timeflies.py create-test-data"
    )
    print(
        "   3. [OK] Data splits created"
        if workflow_complete
        else "   3. [ERROR] Run: python run_timeflies.py setup"
    )
    print(
        "   4. [OK] Verification complete"
        if all_passed and workflow_complete
        else "   4. [ERROR] Fix issues above"
    )
    print("   5. Next: python run_timeflies.py train")
    print("   6. Evaluate: python run_timeflies.py evaluate")
    print("   7. Analyze: python run_timeflies.py analyze")
    print(
        "   WARNING:  Optional: python run_timeflies.py batch-correct (between steps 3-4)"
    )

    print("=" * 60)

    # Restore original logging level and warnings
    logging.getLogger().setLevel(original_level)
    logging.getLogger("timeflies.common.core.active_config").setLevel(
        active_config_level
    )
    logging.getLogger("timeflies.projects.fruitfly_aging.core.config_manager").setLevel(
        config_manager_level
    )
    warnings.resetwarnings()

    return 0 if all_passed else 1


def run_test_suite(args) -> int:
    """Run the test suite using test_runner.py functionality."""
    print("Running TimeFlies Test Suite...")
    print("=" * 60)

    try:
        # Import the test runner function
        sys.path.append(str(Path.cwd() / "tests"))
        from test_runner import run_tests

        # Extract test options from args
        test_type = getattr(args, "test_type", "all")
        coverage = getattr(args, "coverage", False)
        fast = getattr(args, "fast", False)
        debug = getattr(args, "debug", False)
        rerun = getattr(args, "rerun", False)
        verbose = getattr(args, "verbose", False)

        print(f"Test type: {test_type}")
        if fast:
            print("Mode: Fast (unit + integration only)")
        if coverage:
            print("Coverage: Enabled")
        if debug:
            print("Debug: Stop on first failure")
        if rerun:
            print("Re-run: Failed tests only")
        print("")

        # Run the tests
        success = run_tests(
            test_type=test_type,
            verbose=verbose,
            coverage=coverage,
            fast=fast,
            debug=debug,
            rerun_failures=rerun,
        )

        return 0 if success else 1

    except ImportError as e:
        print(f"[ERROR] Could not import test runner: {e}")
        print("NOTE: Make sure tests/test_runner.py exists")
        return 1
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return 1


def new_setup_command(args) -> int:
    """
    Complete setup workflow for users: split data + optional batch correction + verify system.
    For developers: just create environments.
    """
    if hasattr(args, "dev") and args.dev:
        print("SETUP: TimeFlies Developer Setup")
        print("=" * 50)
        print("Setting up development environments...")

        # For devs: just create the environments
        return setup_dev_environments()

    print("LAUNCH: TimeFlies Complete Setup")
    print("=" * 50)
    print("Setting up your TimeFlies environment...")

    # 0. Create user configuration and templates
    print("\n0. Setting up configuration and templates...")
    setup_result = setup_user_environment()
    if setup_result != 0:
        print("[ERROR] User environment setup failed.")
        return setup_result
    print("[OK] Configuration and templates ready")

    # Copy remaining config files now that pre-setup configs are available
    additional_config_result = copy_remaining_config_files()
    if additional_config_result != 0:
        print("WARNING: Some additional config files may not have been copied")

    # 1. Create data splits
    print("\n1. Creating data splits...")
    if hasattr(args, "force_split") and args.force_split:
        print("   FORCE: Force split enabled - will recreate existing splits")
    split_result = split_command(args)
    if split_result != 0:
        print("[ERROR] Data split creation failed.")
        return split_result

    # 2. Optional batch correction
    if hasattr(args, "batch_correct") and args.batch_correct:
        print("\n2. Running batch correction...")
        batch_result = batch_command(args)
        if batch_result != 0:
            print("[ERROR] Batch correction failed.")
            return batch_result
        print("[OK] Batch correction completed")
    else:
        print("\n2. Skipping batch correction")

    # 3. Create output directories
    print("\n3. Creating output directories...")
    from pathlib import Path

    output_dirs = [
        "outputs/fruitfly_aging/experiments/uncorrected",
        "outputs/fruitfly_aging/experiments/batch_corrected",
        "outputs/fruitfly_aging/eda/uncorrected",
        "outputs/fruitfly_aging/eda/batch_corrected",
        "outputs/fruitfly_alzheimers/experiments/uncorrected",
        "outputs/fruitfly_alzheimers/experiments/batch_corrected",
        "outputs/fruitfly_alzheimers/eda/uncorrected",
        "outputs/fruitfly_alzheimers/eda/batch_corrected",
    ]

    # Create directories if they don't exist (skip during tests)
    if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("[OK] Output directories created")
    print(
        "      INFO: outputs/ - All results, plots, logs, and analysis reports organized by project"
    )

    # 4. System verification (always runs last)
    print("\n4. Verifying system setup...")
    from common.cli.system_checks import verify_system

    dev_mode = hasattr(args, "dev") and args.dev
    verify_result = verify_system(dev_mode=dev_mode)
    if not verify_result:  # verify_system returns True/False, not 0/1
        print("[ERROR] System verification failed. Please fix issues above.")
        return 1

    print("\nSUCCESS: SETUP COMPLETE!")
    print("=" * 50)
    print("Your TimeFlies environment is ready!")
    print("\nNext steps:")
    print("  ANALYSIS: Run EDA:        timeflies eda --save-report")
    print("  TRAINING: Train models:   timeflies train")
    print("  EVALUATION: Evaluate:       timeflies evaluate")
    print("  RESEARCH: Analyze:        timeflies analyze")
    print("  WORKFLOW: Full pipeline:  timeflies train --with-eda --with-analysis")
    print("\nAll results will be saved to organized directories in outputs/")

    return 0


def split_command(args) -> int:
    """Create train/eval data splits from original data."""
    try:
        from common.data.setup import DataSetupManager

        print("PROCESS: Creating train/eval data splits...")
        print("============================================================")

        # Use the existing setup manager to create splits
        force_split = hasattr(args, "force_split") and args.force_split
        if force_split:
            print("FORCE: Force split enabled - will recreate existing splits")

        setup_manager = DataSetupManager()
        success = setup_manager.setup_data(force_split=force_split)

        if success:
            print("[OK] Data splits created successfully!")
            return 0
        else:
            print("[ERROR] Failed to create data splits")
            return 1

    except Exception as e:
        print(f"Error creating data splits: {e}")
        return 1


def process_data_splits(
    file_path: Path,
    file_type: str,
    split_size: int,
    encoding_var: str,
    random_state: int,
) -> bool:
    """Process data splits for a single file with enhanced stratification."""
    import anndata
    import numpy as np
    from sklearn.model_selection import train_test_split

    try:
        # Generate output file names
        file_stem = file_path.stem

        # Handle different source file patterns
        if file_stem.endswith("_original_batch"):
            # Input: drosophila_head_aging_original_batch.h5ad
            # Output: drosophila_head_aging_train_batch.h5ad, drosophila_head_aging_eval_batch.h5ad
            base_name = file_stem.replace("_original_batch", "")
        elif file_stem.endswith("_original"):
            # Input: drosophila_head_aging_original.h5ad
            # Output: drosophila_head_aging_train.h5ad, drosophila_head_aging_eval.h5ad
            base_name = file_stem.replace("_original", "")
        else:
            base_name = file_stem

        data_dir = file_path.parent

        if file_type == "batch":
            train_file = data_dir / f"{base_name}_train_batch.h5ad"
            eval_file = data_dir / f"{base_name}_eval_batch.h5ad"
        else:
            train_file = data_dir / f"{base_name}_train.h5ad"
            eval_file = data_dir / f"{base_name}_eval.h5ad"

        # Check if splits already exist
        if train_file.exists() or eval_file.exists():
            print(f"SKIP:  Splits already exist for {file_path.name} - skipping")
            print(
                f"   FOUND: Found: {train_file.name if train_file.exists() else ''} {eval_file.name if eval_file.exists() else ''}"
            )
            return True

        print(f"FILE: Processing: {file_path.name}")

        # Load data
        adata = anndata.read_h5ad(file_path)
        print(f"   DATA: Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Determine stratification columns
        stratify_cols = []

        # Primary stratification: encoding variable (age/disease status)
        if encoding_var in adata.obs.columns:
            stratify_cols.append(encoding_var)
            print(f"   TARGET: Primary stratification: {encoding_var}")

        # Secondary stratification: sex
        sex_cols = ["sex", "Sex", "gender", "Gender"]
        for col in sex_cols:
            if col in adata.obs.columns and col not in stratify_cols:
                stratify_cols.append(col)
                print(f"   SECONDARY:  Secondary stratification: {col}")
                break

        # Tertiary stratification: cell type
        celltype_cols = [
            "cell_type",
            "celltype",
            "Cell_Type",
            "cluster",
            "leiden",
            "louvain",
        ]
        for col in celltype_cols:
            if col in adata.obs.columns and col not in stratify_cols:
                stratify_cols.append(col)
                print(f"   DNA: Tertiary stratification: {col}")
                break

        if not stratify_cols:
            print("   WARNING:  No stratification columns found, using random split")
            stratify_labels = None
        else:
            # Create combined stratification labels
            if len(stratify_cols) == 1:
                stratify_labels = adata.obs[stratify_cols[0]].astype(str)
            else:
                stratify_labels = (
                    adata.obs[stratify_cols].astype(str).agg("_".join, axis=1)
                )

            print(f"   METRICS: Stratification groups: {len(stratify_labels.unique())}")

        # Calculate split ratio to get desired eval size
        total_cells = adata.shape[0]
        if split_size >= total_cells:
            print(
                f"   WARNING:  Split size ({split_size}) >= total cells ({total_cells}), using 20% split"
            )
            test_size = 0.2
        else:
            test_size = split_size / total_cells

        print(
            f"   ✂️  Split ratio: {test_size:.3f} ({int(total_cells * test_size)} eval cells)"
        )

        # Perform stratified split
        indices = np.arange(adata.shape[0])

        if stratify_labels is not None:
            train_idx, eval_idx = train_test_split(
                indices,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=random_state,
            )
        else:
            train_idx, eval_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )

        # Create splits
        adata_train = adata[train_idx].copy()
        adata_eval = adata[eval_idx].copy()

        # Save splits
        print(f"   SAVE: Saving: {train_file.name} ({len(adata_train)} cells)")
        adata_train.write_h5ad(train_file)

        print(f"   SAVE: Saving: {eval_file.name} ({len(adata_eval)} cells)")
        adata_eval.write_h5ad(eval_file)

        print("   [OK] Split completed successfully")
        return True

    except Exception as e:
        print(f"   [ERROR] Error processing {file_path.name}: {e}")
        return False


def eda_command(args, config) -> int:
    """Run exploratory data analysis on the dataset."""
    import os
    from pathlib import Path

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING

    try:
        print("DATA: Starting EDA with project settings:")
        print(f"   Project: {getattr(config, 'project', 'unknown')}")
        print(f"   Tissue: {config.data.tissue}")
        print(f"   Batch corrected: {getattr(args, 'batch_corrected', False)}")
        print(f"   Split: {getattr(args, 'split', 'all')}")

        # Apply CLI overrides to config
        if hasattr(args, "batch_corrected") and args.batch_corrected:
            config.data.batch_correction.enabled = True
        if hasattr(args, "tissue") and args.tissue:
            config.data.tissue = args.tissue

        # Create EDA output directory structure
        project = getattr(config, "project", "fruitfly_alzheimers")
        tissue = config.data.tissue
        correction = (
            "batch_corrected" if config.data.batch_correction.enabled else "uncorrected"
        )

        # EDA analyzes full dataset - simple path structure
        eda_dir = Path(f"outputs/{project}/eda/{correction}/{tissue}")
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            eda_dir.mkdir(parents=True, exist_ok=True)

        # Initialize EDA handler with output directory
        eda_handler = EDAHandler(config, output_dir=str(eda_dir))

        # Run comprehensive EDA on full dataset
        eda_handler.run_comprehensive_eda()

        # Generate HTML report if requested
        if hasattr(args, "save_report") and args.save_report:
            report_path = eda_dir / "eda_report.html"
            eda_handler.generate_html_report(report_path)
            print(f"   DOC: HTML report saved to: {report_path}")

        print("\n[OK] EDA completed successfully!")
        print(f"   Results saved to: {eda_dir}")
        return 0

    except Exception as e:
        print(f"[ERROR] EDA failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def train_command(args, config) -> int:
    """Train a model using project configuration settings."""
    try:
        # Run EDA first if requested
        if hasattr(args, "with_eda") and args.with_eda:
            print("\nDATA: Running EDA before training...")
            result = eda_command(args, config)
            if result != 0:
                print("[ERROR] EDA failed, stopping pipeline")
                return result

        print("Starting training with project settings:")
        print(f"   Project: {getattr(config, 'project', 'unknown')}")
        print(f"   Tissue: {config.data.tissue}")
        print(f"   Model: {config.data.model}")
        print(f"   Target: {config.data.target_variable}")
        print(f"   Batch correction: {config.data.batch_correction.enabled}")

        # Use common PipelineManager for all projects
        from common.core import PipelineManager

        # Initialize and run pipeline
        pipeline = PipelineManager(config)
        results = pipeline.run()

        print("\n[OK] Training completed successfully!")

        # Run analysis after training if requested
        if hasattr(args, "with_analysis") and args.with_analysis:
            print("\nRESEARCH: Running analysis after training...")
            result = analyze_command(args, config)
            if result != 0:
                print("WARNING:  Analysis failed but training was successful")

        # Check if model was actually saved (based on validation loss improvement)
        model_path = results.get("model_path", "outputs/models/")
        import os

        model_file = os.path.join(model_path, "best_model.h5")

        # Check if model was updated by comparing file modification time with start time
        if os.path.exists(model_file):
            import time

            file_mod_time = os.path.getmtime(model_file)
            # If file was modified in the last 5 minutes, it was likely updated
            if time.time() - file_mod_time < 300:
                print(f"   [OK] Model saved (validation loss improved): {model_path}")
            else:
                print("   SKIP:  Model not saved (validation loss did not improve)")
                print(f"   FOUND: Existing model location: {model_path}")
        else:
            print(f"   [OK] New model saved: {model_path}")

        print(
            f"   DATA: Results saved to: {results.get('results_path', 'outputs/results/')}"
        )

        # Show best results path only if model improved
        if results.get("model_improved", False) and "best_results_path" in results:
            print(
                f"   BEST: Best results also saved to: {results['best_results_path']}"
            )

        if "duration" in results:
            print(f"   TIME:  Training duration: {results['duration']:.1f} seconds")

        return 0

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def batch_command(args) -> int:
    """Run batch correction pipeline."""
    if not BATCH_CORRECTION_AVAILABLE:
        print("Error: Cannot perform batch correction.")
        print("The scVI dependencies are not installed in this environment.")
        print(
            "To perform batch correction, install: pip install scvi-tools scanpy scib"
        )
        print("Note: You can still use existing batch-corrected data files.")
        print(
            "Make sure you're in the batch correction environment: source activate_batch.sh"
        )
        return 1

    print(f"Running batch correction for {args.tissue} tissue...")

    try:
        # Create batch corrector with new data structure
        # Determine project from config or use default
        project = "fruitfly_aging"  # Default project
        base_dir = f"data/{project}"

        # Initialize batch corrector with updated paths
        # IMPORTANT: Never use original file for batch correction to avoid data leakage
        if args.original:
            print(
                "WARNING: Using --original flag for batch correction can cause data leakage!"
            )
            print(
                "Batch correction should only be trained on training data, not the full dataset."
            )
            print("Switching to train/eval mode for proper ML hygiene.")
            use_original = False
        else:
            use_original = False

        batch_corrector = BatchCorrector(
            data_type="uncorrected",
            tissue=args.tissue,
            base_dir=base_dir,
            use_original=use_original,
        )

        # Run the batch correction training
        print("Training scVI model on TRAINING data only...")
        print("Then applying trained model to evaluation data (query mode)...")
        print("This prevents data leakage into the holdout set.")
        batch_corrector.run()

        # Run evaluation if requested
        if args.evaluate:
            print("\nEvaluating batch correction quality...")
            batch_corrector.evaluate_metrics()

        # Generate visualizations if requested
        if args.visualize:
            print("\nGenerating UMAP visualizations...")
            batch_corrector.umap_visualization()

        print("\n" + "=" * 60)
        print("✓ BATCH CORRECTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show what was created
        tissue = args.tissue
        print(f"✓ scVI model trained on: drosophila_{tissue}_aging_train.h5ad")
        print(f"✓ Model applied to: drosophila_{tissue}_aging_eval.h5ad (query mode)")
        print(f"✓ Created: drosophila_{tissue}_aging_train_batch.h5ad")
        print(f"✓ Created: drosophila_{tissue}_aging_eval_batch.h5ad")
        print("✓ Original file untouched (prevents data leakage)")

        print("\nNext steps:")
        print("1. Return to main environment: source activate.sh")
        print("2. Train with batch data:     python run_timeflies.py train")
        print("   (Will automatically use batch-corrected files)")
        print(
            "\nNote: No additional setup needed - batch correction used existing splits!"
        )

        return 0

    except Exception as e:
        print(f"Batch correction failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def evaluate_command(args, config) -> int:
    """Evaluate a trained model using project configuration settings."""
    # Suppress TensorFlow warnings for cleaner output
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING

    try:
        # Run EDA first if requested
        if hasattr(args, "with_eda") and args.with_eda:
            print("\nDATA: Running EDA before evaluation...")
            result = eda_command(args, config)
            if result != 0:
                print("[ERROR] EDA failed, stopping pipeline")
                return result

        print("Starting evaluation with project settings:")
        print(f"   Project: {getattr(config, 'project', 'unknown')}")
        print(f"   Tissue: {config.data.tissue}")
        print(f"   Model: {config.data.model}")
        print(f"   Target: {config.data.target_variable}")

        # Handle CLI flag overrides for SHAP and visualizations
        if hasattr(args, "interpret") and args.interpret:
            print("   DATA: SHAP interpretation: ENABLED (via --interpret flag)")
            # Temporarily override config
            if hasattr(config, "interpretation") and hasattr(
                config.interpretation, "shap"
            ):
                original_shap = config.interpretation.shap.enabled
                config.interpretation.shap.enabled = True
            else:
                # Create the config structure if it doesn't exist
                if not hasattr(config, "interpretation"):
                    from types import SimpleNamespace

                    config.interpretation = SimpleNamespace()
                    config.interpretation.shap = SimpleNamespace()
                config.interpretation.shap.enabled = True
                original_shap = False
        else:
            original_shap = None

        if hasattr(args, "visualize") and args.visualize:
            print("   METRICS: Visualizations: ENABLED (via --visualize flag)")
            # Temporarily override config
            if hasattr(config, "visualizations"):
                original_viz = config.visualizations.enabled
                config.visualizations.enabled = True
            else:
                from types import SimpleNamespace

                config.visualizations = SimpleNamespace()
                config.visualizations.enabled = True
                original_viz = False
        else:
            original_viz = None

        # Use common PipelineManager for all projects
        from common.core import PipelineManager

        # Initialize pipeline and run evaluation-only workflow
        # (This includes metrics, interpretation, and visualizations based on config)
        pipeline = PipelineManager(config)
        pipeline.run_evaluation()

        # Restore original config settings if overridden
        if original_shap is not None:
            config.interpretation.shap.enabled = original_shap
        if original_viz is not None:
            config.visualizations.enabled = original_viz

        print("\n[OK] Evaluation completed successfully!")

        # Run analysis after evaluation if requested
        if hasattr(args, "with_analysis") and args.with_analysis:
            print("\nRESEARCH: Running analysis after evaluation...")
            result = analyze_command(args, config)
            if result != 0:
                print("WARNING:  Analysis failed but evaluation was successful")

        return 0

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def analyze_command(args, config) -> int:
    """Run project-specific analysis on a trained model."""
    # Suppress TensorFlow warnings for cleaner output
    import os
    from pathlib import Path

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING

    try:
        print("RESEARCH: Starting analysis with project settings:")
        print(f"   Project: {getattr(config, 'project', 'unknown')}")
        print(f"   Tissue: {config.data.tissue}")
        print(f"   Model: {config.data.model}")
        print(f"   Target: {config.data.target_variable}")

        # Store custom analysis script path in config for pipeline manager
        if hasattr(args, "analysis_script") and args.analysis_script:
            config._custom_analysis_script = args.analysis_script
            print(f"   Custom script: {args.analysis_script}")

        # Check for CLI-provided predictions path first
        predictions_path = None
        if hasattr(args, "predictions_path") and args.predictions_path:
            predictions_path = Path(args.predictions_path)
            print(f"FILE: Using provided predictions path: {predictions_path}")
        else:
            # Auto-detect predictions in new experiment structure
            from common.utils.path_manager import PathManager

            path_manager = PathManager(config)

            # Try to find best model for current config
            try:
                best_model_dir = path_manager.get_best_model_dir_for_config()
                predictions_path = (
                    Path(best_model_dir) / "evaluation" / "predictions.csv"
                )
                if predictions_path.exists():
                    print(f"[OK] Found predictions from best model: {predictions_path}")
                else:
                    predictions_path = None
            except Exception:
                predictions_path = None

        if predictions_path and predictions_path.exists():
            print(f"[OK] Found existing predictions at {predictions_path}")
            print("DATA: Running analysis on existing predictions...")

            # Just run the analysis script without reloading everything
            from common.core import PipelineManager

            pipeline = PipelineManager(config)

            # Only run the analysis script part
            if hasattr(pipeline, "run_analysis_script"):
                pipeline.run_analysis_script()

            print("\n[OK] Analysis completed successfully!")
            return 0

        print("WARNING:  No predictions found, need to generate them...")

        # Check if model exists
        from common.utils.path_manager import PathManager

        path_manager = PathManager(config)
        model_dir = path_manager.construct_model_directory()
        model_path = Path(model_dir) / "model.h5"

        if not model_path.exists():
            print(f"WARNING:  No trained model found at {model_path}")
            print("PACKAGE: Training model first...")

            # Run training
            from common.core import PipelineManager

            pipeline = PipelineManager(config)
            pipeline.load_or_train_model()
            print("[OK] Model training complete!")

        # Enable analysis script execution in config
        if not hasattr(config.analysis, "run_analysis_script"):
            print("[ERROR] Analysis script configuration not found in config")
            return 1

        # Temporarily enable analysis script for this command
        original_enabled = getattr(
            config.analysis.run_analysis_script, "enabled", False
        )
        config.analysis.run_analysis_script.enabled = True

        try:
            # Use common PipelineManager to load model and run analysis
            from common.core import PipelineManager

            # Initialize pipeline and run evaluation with analysis
            pipeline = PipelineManager(config)
            pipeline.run_evaluation()

            print("\n[OK] Analysis completed successfully!")
            return 0

        finally:
            # Restore original setting
            config.analysis.run_analysis_script.enabled = original_enabled

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def create_test_data_command(args) -> int:
    """Create test data fixtures using 3-tier strategy: tiny real + synthetic + dev real data."""
    try:
        import json
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import scanpy as sc

        tier = getattr(args, "tier", "all")
        print(f"TESTING: Creating Test Data Fixtures - Tier: {tier}")
        print("=" * 60)
        print("3-Tier Strategy:")
        print("  PACKAGE: Tiny: Small real samples (committed to git)")
        print("  AUTO: Synthetic: Generated from metadata")
        print("  RESEARCH: Real: Full developer samples (gitignored)")
        print("")

        # For synthetic and tiny tiers, we can work from existing metadata
        if tier in ["synthetic", "tiny"]:
            print("SEARCH: Looking for existing metadata...")
            return create_from_metadata(tier, args)

        # For real tier, we need actual data files
        data_root = Path("data")
        if not data_root.exists():
            print(
                "[ERROR] Data directory not found. Place data files in data/[project]/[tissue]/ first."
            )
            print(
                "NOTE: For synthetic data: run with --tier synthetic (uses existing metadata)"
            )
            return 1

        projects_found = []
        results = []

        # Scan for project directories
        for project_dir in data_root.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue

            for tissue_dir in project_dir.iterdir():
                if not tissue_dir.is_dir():
                    continue

                h5ad_files = list(tissue_dir.glob("*.h5ad"))
                if not h5ad_files:
                    continue

                # Find best source file
                original_files = [f for f in h5ad_files if "original" in f.name]
                train_files = [
                    f for f in h5ad_files if "train" in f.name and "batch" not in f.name
                ]

                if original_files:
                    data_file = original_files[0]
                elif train_files:
                    data_file = train_files[0]
                else:
                    continue

                print(f"\nFOUND: Processing: {project_dir.name}/{tissue_dir.name}")
                print(f"DATA: Source file: {data_file.name}")

                # Create test data based on tier
                if tier in ["tiny", "all"]:
                    result = create_tiny_fixtures(
                        project_dir.name, tissue_dir.name, data_file
                    )
                    if result:
                        results.append(result)

                if tier in ["synthetic", "all"]:
                    result = create_synthetic_fixtures(
                        project_dir.name, tissue_dir.name, data_file
                    )
                    if result:
                        results.append(result)

                if tier in ["real", "all"]:
                    result = create_real_fixtures(
                        project_dir.name, tissue_dir.name, data_file
                    )
                    if result:
                        results.append(result)

                projects_found.append(f"{project_dir.name}/{tissue_dir.name}")

        if not results:
            print("\n[ERROR] No test data created.")
            return 1

        # Save summary
        summary = {
            "created_at": pd.Timestamp.now().isoformat(),
            "tier": tier,
            "projects": results,
            "total_projects": len(projects_found),
        }

        summary_path = Path("tests/fixtures/test_data_summary.json")
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSUCCESS: Test data created for {len(projects_found)} project(s)")
        print(f"DOC: Summary saved to: {summary_path}")

        return 0

    except ImportError as e:
        print(f"[ERROR] Required dependencies not available: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to create test data: {e}")
        return 1


def create_from_metadata(tier, args) -> int:
    """Create test data from existing metadata files."""
    from pathlib import Path

    fixtures_root = Path("tests/fixtures")
    if not fixtures_root.exists():
        print("[ERROR] No test fixtures directory found")
        return 1

    results = []

    # Scan for existing metadata
    for project_dir in fixtures_root.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue

        print(f"\nFOUND: Processing project: {project_dir.name}")

        # Look for existing metadata files
        metadata_files = list(project_dir.glob("*_stats.json"))
        for metadata_file in metadata_files:
            # Extract tissue name from filename
            # test_data_head_stats.json -> head
            tissue = metadata_file.stem.replace("test_data_", "").replace("_stats", "")

            print(f"DATA: Found metadata: {metadata_file.name} -> tissue: {tissue}")

            if tier == "tiny":
                result = create_tiny_from_metadata(
                    project_dir.name, tissue, metadata_file, args
                )
                if result:
                    results.append(result)

            elif tier == "synthetic":
                result = create_synthetic_from_metadata(
                    project_dir.name, tissue, metadata_file, args
                )
                if result:
                    results.append(result)

    if not results:
        print(f"[ERROR] No metadata found for {tier} data creation")
        return 1

    print(f"\nSUCCESS: Created {len(results)} {tier} fixtures!")
    return 0


def load_test_data_config():
    """Load test data defaults from config (dev-only, not shipped to users)."""
    from pathlib import Path

    import yaml

    config_path = Path("configs/test_data_defaults.yaml")
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception:
            pass

    # Fallback defaults if config not found
    return {
        "test_data": {
            "defaults": {
                "tiny": {"cells": 50, "genes": 100},
                "synthetic": {"cells": 500, "genes": 1000},
                "real": {"cells": 5000, "genes": 2000},
            },
            "batch_correction": {"noise_std": 0.1},
            "random_seed": 42,
        }
    }


def create_tiny_from_metadata(project, tissue, metadata_file, args, seed=None):
    """Create tiny fixtures from metadata."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # Load config defaults
    config = load_test_data_config()
    defaults = config["test_data"]["defaults"]["tiny"]
    if seed is None:
        seed = config["test_data"]["random_seed"]

    np.random.seed(seed)

    try:
        print("  PACKAGE: Creating tiny fixture from metadata...")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get size parameters (allow override, fallback to config)
        n_cells = getattr(args, "cells", None) or defaults["cells"]
        n_genes = getattr(args, "genes", None) or defaults["genes"]
        batch_versions = getattr(args, "batch_versions", False)

        # Generate expression data matching real patterns
        expr_stats = metadata.get("expression_stats", {})
        synthetic_data = np.random.lognormal(
            mean=np.log(expr_stats.get("non_zero_mean", 1.0)),
            sigma=1.0,
            size=(n_cells, n_genes),
        )

        # Apply sparsity
        sparsity = expr_stats.get("sparsity", 0.8)
        mask = np.random.random((n_cells, n_genes)) < sparsity
        synthetic_data[mask] = 0

        # Create synthetic metadata matching real patterns
        obs_data = {}
        if "age_distribution" in metadata:
            ages = list(metadata["age_distribution"].keys())
            obs_data["age"] = np.random.choice(ages, n_cells)

        if "sex_distribution" in metadata:
            sexes = list(metadata["sex_distribution"].keys())
            obs_data["sex"] = np.random.choice(sexes, n_cells)

        obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        # Create AnnData
        import anndata

        adata_tiny = anndata.AnnData(X=synthetic_data, obs=obs_df, var=var_df)

        # Save fixtures
        output_dir = Path("tests/fixtures") / project

        # Regular version
        tiny_path = output_dir / f"tiny_{tissue}.h5ad"
        adata_tiny.write_h5ad(tiny_path)
        files_created = [f"tiny_{tissue}.h5ad"]

        # Batch corrected version (if requested)
        if batch_versions:
            # Simulate batch correction using config parameters
            noise_std = config["test_data"]["batch_correction"]["noise_std"]
            batch_data = synthetic_data.copy()
            batch_noise = np.random.normal(0, noise_std, batch_data.shape)
            batch_data[batch_data > 0] += batch_noise[batch_data > 0]
            batch_data = np.maximum(batch_data, 0)  # Keep non-negative

            adata_batch = anndata.AnnData(X=batch_data, obs=obs_df, var=var_df)

            batch_path = output_dir / f"tiny_{tissue}_batch.h5ad"
            adata_batch.write_h5ad(batch_path)
            files_created.append(f"tiny_{tissue}_batch.h5ad")

        print(
            f"    [OK] Tiny: {', '.join(files_created)} ({n_cells} cells, {n_genes} genes)"
        )
        return {"tier": "tiny", "project": project, "tissue": tissue}

    except Exception as e:
        print(f"    [ERROR] Tiny creation failed: {e}")
        return None


def create_synthetic_from_metadata(project, tissue, metadata_file, args, seed=None):
    """Create synthetic fixtures from metadata."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # Load config defaults
    config = load_test_data_config()
    defaults = config["test_data"]["defaults"]["synthetic"]
    if seed is None:
        seed = config["test_data"]["random_seed"]

    np.random.seed(seed)

    try:
        print("  AUTO: Creating synthetic fixture from metadata...")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get size parameters (allow override, fallback to config)
        n_cells = getattr(args, "cells", None) or defaults["cells"]
        n_genes = getattr(args, "genes", None) or defaults["genes"]
        batch_versions = getattr(args, "batch_versions", False)

        # Generate expression data matching real patterns
        expr_stats = metadata.get("expression_stats", {})
        synthetic_data = np.random.lognormal(
            mean=np.log(expr_stats.get("non_zero_mean", 1.0)),
            sigma=1.0,
            size=(n_cells, n_genes),
        )

        # Apply sparsity
        sparsity = expr_stats.get("sparsity", 0.8)
        mask = np.random.random((n_cells, n_genes)) < sparsity
        synthetic_data[mask] = 0

        # Create synthetic metadata
        obs_data = {}
        if "age_distribution" in metadata:
            ages = list(metadata["age_distribution"].keys())
            obs_data["age"] = np.random.choice(ages, n_cells)

        if "sex_distribution" in metadata:
            sexes = list(metadata["sex_distribution"].keys())
            obs_data["sex"] = np.random.choice(sexes, n_cells)

        obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        # Create AnnData
        import anndata

        adata_synthetic = anndata.AnnData(X=synthetic_data, obs=obs_df, var=var_df)

        # Save fixtures
        output_dir = Path("tests/fixtures") / project

        # Regular version
        synthetic_path = output_dir / f"synthetic_{tissue}.h5ad"
        adata_synthetic.write_h5ad(synthetic_path)
        files_created = [f"synthetic_{tissue}.h5ad"]

        # Batch corrected version (if requested)
        if batch_versions:
            # Simulate batch correction using config parameters
            noise_std = config["test_data"]["batch_correction"]["noise_std"]
            batch_data = synthetic_data.copy()
            batch_noise = np.random.normal(0, noise_std, batch_data.shape)
            batch_data[batch_data > 0] += batch_noise[batch_data > 0]
            batch_data = np.maximum(batch_data, 0)

            adata_batch = anndata.AnnData(X=batch_data, obs=obs_df, var=var_df)

            batch_path = output_dir / f"synthetic_{tissue}_batch.h5ad"
            adata_batch.write_h5ad(batch_path)
            files_created.append(f"synthetic_{tissue}_batch.h5ad")

        print(
            f"    [OK] Synthetic: {', '.join(files_created)} ({n_cells} cells, {n_genes} genes)"
        )
        return {"tier": "synthetic", "project": project, "tissue": tissue}

    except Exception as e:
        print(f"    [ERROR] Synthetic creation failed: {e}")
        return None


def create_tiny_fixtures(project, tissue, data_file, seed=42):
    """Create tiny real data fixtures (committed to git) - Tier 1."""
    import json
    from pathlib import Path

    import numpy as np
    import scanpy as sc

    np.random.seed(seed)

    try:
        print("  PACKAGE: Creating tiny fixtures...")
        adata = sc.read_h5ad(data_file)

        # Very small samples - suitable for git
        n_cells = min(50, adata.n_obs)
        n_genes = min(100, adata.n_vars)

        cell_indices = np.random.choice(adata.n_obs, n_cells, replace=False)
        gene_indices = np.random.choice(adata.n_vars, n_genes, replace=False)

        adata_tiny = adata[cell_indices, gene_indices].copy()

        # Save to fixtures directory
        output_dir = Path("tests/fixtures") / project
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            output_dir.mkdir(parents=True, exist_ok=True)

        tiny_path = output_dir / f"tiny_{tissue}.h5ad"
        adata_tiny.write_h5ad(tiny_path)

        # Generate and save metadata
        stats = generate_data_stats(adata_tiny, project, tissue, tier="tiny")
        metadata_path = output_dir / f"tiny_{tissue}_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"    [OK] Tiny: {tiny_path} ({n_cells} cells, {n_genes} genes)")

        return {"tier": "tiny", "project": project, "tissue": tissue, "stats": stats}

    except Exception as e:
        print(f"    [ERROR] Tiny fixtures failed: {e}")
        return None


def create_synthetic_fixtures(project, tissue, data_file, seed=42):
    """Create synthetic data from metadata - Tier 2."""
    import json
    from pathlib import Path

    import numpy as np

    np.random.seed(seed)

    try:
        print("  AUTO: Creating synthetic fixtures...")

        # First check if we have existing metadata to use
        output_dir = Path("tests/fixtures") / project
        metadata_path = output_dir / f"tiny_{tissue}_metadata.json"

        if metadata_path.exists():
            # Use existing metadata to generate synthetic data
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Generate synthetic data matching the patterns
            n_cells = 500  # Medium size
            n_genes = 1000

            # Create synthetic expression matrix based on real stats
            expr_stats = metadata.get("expression_stats", {})

            # Generate data matching real distributions
            synthetic_data = np.random.lognormal(
                mean=np.log(expr_stats.get("non_zero_mean", 1.0)),
                sigma=1.0,
                size=(n_cells, n_genes),
            )

            # Apply sparsity pattern
            sparsity = expr_stats.get("sparsity", 0.8)
            mask = np.random.random((n_cells, n_genes)) < sparsity
            synthetic_data[mask] = 0

            # Create synthetic AnnData object
            import pandas as pd

            # Create synthetic metadata matching real patterns
            obs_data = {}
            if "age_distribution" in metadata:
                ages = list(metadata["age_distribution"].keys())
                obs_data["age"] = np.random.choice(ages, n_cells)

            if "sex_distribution" in metadata:
                sexes = list(metadata["sex_distribution"].keys())
                obs_data["sex"] = np.random.choice(sexes, n_cells)

            obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
            var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

            # Create AnnData
            import anndata

            adata_synthetic = anndata.AnnData(X=synthetic_data, obs=obs_df, var=var_df)

            # Save synthetic data
            synthetic_path = output_dir / f"synthetic_{tissue}.h5ad"
            adata_synthetic.write_h5ad(synthetic_path)

            print(
                f"    [OK] Synthetic: {synthetic_path} ({n_cells} cells, {n_genes} genes)"
            )

            return {
                "tier": "synthetic",
                "project": project,
                "tissue": tissue,
                "size": (n_cells, n_genes),
            }

        else:
            print("    WARNING:  No metadata found for synthetic generation")
            return None

    except Exception as e:
        print(f"    [ERROR] Synthetic fixtures failed: {e}")
        return None


def create_real_fixtures(project, tissue, data_file, seed=42):
    """Create full-scale real data fixtures (gitignored) - Tier 3."""
    from pathlib import Path

    import numpy as np
    import scanpy as sc

    np.random.seed(seed)

    try:
        print("  RESEARCH: Creating real fixtures...")
        adata = sc.read_h5ad(data_file)

        # Larger realistic samples for thorough testing
        n_cells = min(5000, adata.n_obs)
        n_genes = min(2000, adata.n_vars)

        cell_indices = np.random.choice(adata.n_obs, n_cells, replace=False)
        gene_indices = np.random.choice(adata.n_vars, n_genes, replace=False)

        adata_real = adata[cell_indices, gene_indices].copy()

        # Save to fixtures directory (will be gitignored)
        output_dir = Path("tests/fixtures") / project
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            output_dir.mkdir(parents=True, exist_ok=True)

        real_path = output_dir / f"real_{tissue}.h5ad"
        adata_real.write_h5ad(real_path)

        print(f"    [OK] Real: {real_path} ({n_cells} cells, {n_genes} genes)")

        return {
            "tier": "real",
            "project": project,
            "tissue": tissue,
            "size": (n_cells, n_genes),
        }

    except Exception as e:
        print(f"    [ERROR] Real fixtures failed: {e}")
        return None


def generate_data_stats(adata, project, tissue, batch_corrected=False, tier=None):
    """Generate comprehensive statistics for test data."""
    import numpy as np
    import pandas as pd

    # Basic info
    stats = {
        "project": project,
        "tissue": tissue,
        "tier": tier,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "batch_corrected": batch_corrected,
        "created_at": pd.Timestamp.now().isoformat(),
    }

    # Metadata distributions if available
    if "age" in adata.obs.columns:
        stats["age_distribution"] = adata.obs["age"].value_counts().to_dict()

    if "sex" in adata.obs.columns:
        stats["sex_distribution"] = adata.obs["sex"].value_counts().to_dict()

    # Cell type distribution (try different column names)
    cell_type_cols = ["afca_annotation_broad", "cell_type", "celltype", "annotation"]
    for col in cell_type_cols:
        if col in adata.obs.columns:
            stats["cell_types"] = adata.obs[col].value_counts().head(10).to_dict()
            break

    # Expression data statistics
    if hasattr(adata.X, "toarray"):
        X_array = adata.X.toarray()
    else:
        X_array = adata.X

    stats["expression_stats"] = {
        "min": float(np.min(X_array)),
        "max": float(np.max(X_array)),
        "mean": float(np.mean(X_array)),
        "median": float(np.median(X_array)),
        "sparsity": float(np.mean(X_array == 0)),
        "non_zero_mean": float(np.mean(X_array[X_array > 0]))
        if np.any(X_array > 0)
        else 0.0,
    }

    # Gene info if available
    if hasattr(adata.var, "columns"):
        stats["gene_info"] = {
            "columns": list(adata.var.columns),
            "n_highly_variable": int(
                adata.var.get(
                    "highly_variable", pd.Series([False] * adata.n_vars)
                ).sum()
            ),
        }

    return stats


def setup_dev_environments() -> int:
    """Set up complete development environments with all dependencies."""
    from pathlib import Path

    try:
        print("SETUP:  Setting up TimeFlies development environments...")
        print("=" * 60)

        # Check Python version (same requirement as user install)
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
            print("Install: sudo apt install python3.12 python3.12-venv")
            return 1

        # Create main environment
        print("\nPYTHON: Setting up main environment (.venv)...")
        if Path(".venv").exists():
            print("SKIP:  Removing existing main environment...")
            subprocess.run(["rm", "-rf", ".venv"], check=True)

        subprocess.run([python_cmd, "-m", "venv", ".venv"], check=True)
        print("[OK] Main environment created")

        # Install main dependencies
        print("PACKAGE: Installing main dependencies...")
        venv_pip = ".venv/bin/pip"

        subprocess.run([venv_pip, "install", "--upgrade", "pip"], check=True)
        subprocess.run([venv_pip, "install", "-e", "."], check=True)

        # Install development dependencies from pyproject.toml
        print("SETUP:  Installing development dependencies...")
        subprocess.run([venv_pip, "install", "-e", ".[dev]"], check=True)
        print("[OK] Main and development dependencies installed")

        # Create batch environment
        print("\nRESEARCH: Setting up batch correction environment (.venv_batch)...")
        if Path(".venv_batch").exists():
            print("SKIP:  Removing existing batch environment...")
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

        # Create directory structure
        print("\nFOUND: Creating directory structure...")
        create_project_directories()
        print("[OK] Project directories created")

        print("\nSUCCESS: Development setup complete!")
        print("=" * 60)
        print("Next steps:")
        print("  source .activate.sh         # Activate main environment")
        print("  source .activate_batch.sh   # Switch to batch environment")
        print("  timeflies verify            # Verify setup")
        print("  timeflies test              # Run full test suite")

        return 0

    except Exception as e:
        print(f"[ERROR] Development setup failed: {e}")
        return 1


def create_project_directories():
    """Create necessary project directories for development."""
    import os
    from pathlib import Path

    # Skip directory creation during tests or CI
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
        return

    directories = [
        "data/fruitfly_aging/head",
        "data/fruitfly_alzheimers/head",
        "outputs/fruitfly_aging",
        "outputs/fruitfly_alzheimers",
        "models",
        "coverage",
    ]

    # Create directories if they don't exist (skip during tests)
    if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def setup_user_environment():
    """Create user configuration and templates."""
    import shutil
    from pathlib import Path

    try:
        # Create configs/ directory and copy all config files
        configs_dir = Path("configs")

        # Create directory if it doesn't exist
        if not configs_dir.exists():
            print("   CONFIG: Creating configs/ directory...")
            if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
                configs_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("   CONFIG: configs/ directory already exists")

        # Find source configs from TimeFlies installation
        source_configs_dirs = [
            Path.cwd()
            / ".timeflies_src"
            / "configs",  # user installation (current directory)
            Path(__file__).parent.parent.parent.parent / "configs",  # repo structure
            Path(__file__).parent.parent.parent / "configs",  # installed structure
            Path.home() / ".timeflies_src" / "configs",  # user installation (home)
        ]

        # Check for missing config files and copy them
        # Only copy setup configs before setup runs - others get copied during setup
        pre_setup_configs = [
            "setup.yaml",
            "batch_correction.yaml",
        ]
        missing_configs = [
            cfg for cfg in pre_setup_configs if not (configs_dir / cfg).exists()
        ]

        if missing_configs:
            print(f"   CONFIG: Found {len(missing_configs)} missing config files")
            for source_configs_dir in source_configs_dirs:
                if source_configs_dir.exists():
                    print("      CONFIG: Copying missing configuration files...")
                    copied_any = False
                    for config_name in missing_configs:
                        source_config = source_configs_dir / config_name
                        if source_config.exists():
                            shutil.copy2(source_config, configs_dir / config_name)
                            print(f"         [OK] {config_name}")
                            copied_any = True
                    if copied_any:
                        break
            else:
                print("      [ERROR] Could not find default config templates")
                return 1
        else:
            print("   CONFIG: All required config files present")

        # Remove any old config.yaml in root - we now use configs/ directory
        root_config = Path("config.yaml")
        if root_config.exists():
            print("   CLEANUP: Moving old config.yaml to configs/user_config.yaml")
            user_config = configs_dir / "user_config.yaml"
            if not user_config.exists():
                shutil.move(root_config, user_config)
            else:
                root_config.unlink()  # Remove if user_config already exists

        # Create templates directory if it doesn't exist
        templates_dir = Path("templates")
        if not templates_dir.exists():
            print("   DOC: Creating templates directory...")
            # Create directory if it doesn't exist (skip during tests)
            if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
                templates_dir.mkdir(parents=True, exist_ok=True)

            # Find source templates from TimeFlies installation
            source_templates_dirs = [
                Path(__file__).parent.parent.parent.parent
                / "templates",  # repo structure
                Path(__file__).parent.parent.parent
                / "templates",  # installed structure
            ]

            for source_templates_dir in source_templates_dirs:
                if source_templates_dir.exists():
                    print("      FILE: Copying analysis templates...")
                    # Copy all template files
                    for template_file in source_templates_dir.glob("*"):
                        if template_file.is_file():
                            shutil.copy2(
                                template_file, templates_dir / template_file.name
                            )
                            print(f"         [OK] {template_file.name}")
                    break
            else:
                print("      WARNING:  Could not find templates directory")
                print(
                    "      INFO:  You can create custom analysis scripts in templates/ manually"
                )
        else:
            print("   DOC: templates/ directory already exists")

        # Copy TimeFlies GUI Launcher
        launcher_file = Path("TimeFlies_Launcher.py")
        if not launcher_file.exists():
            print("   GUI: Setting up TimeFlies GUI Launcher...")
            # Find source launcher from TimeFlies installation
            source_launchers = [
                Path(__file__).parent.parent.parent.parent
                / "TimeFlies_Launcher.py",  # repo structure
                Path(__file__).parent.parent.parent
                / "TimeFlies_Launcher.py",  # installed structure
            ]

            for source_launcher in source_launchers:
                if source_launcher.exists():
                    shutil.copy2(source_launcher, launcher_file)
                    print("      [OK] TimeFlies GUI Launcher available")
                    print(
                        "      INFO: Run 'python TimeFlies_Launcher.py' for graphical interface"
                    )
                    break
            else:
                print("      WARNING: Could not find GUI launcher")
                print("      INFO: Use command-line interface instead")
        else:
            print("   GUI: TimeFlies_Launcher.py already exists")

        return 0

    except Exception as e:
        print(f"   [ERROR] Setup failed: {e}")
        return 1


def copy_remaining_config_files():
    """Copy the remaining config files during setup process."""
    import shutil
    from pathlib import Path

    try:
        configs_dir = Path("configs")
        if not configs_dir.exists():
            configs_dir.mkdir(parents=True, exist_ok=True)

        # Configs that get copied during setup (not before)
        setup_configs = [
            "default.yaml",
            "hyperparameter_tuning.yaml",
            "model_queue.yaml",
        ]

        # Find source configs
        source_configs_dirs = [
            Path.cwd() / ".timeflies_src" / "configs",
            Path(__file__).parent.parent.parent.parent / "configs",
            Path(__file__).parent.parent.parent / "configs",
            Path.home() / ".timeflies_src" / "configs",
        ]

        missing_configs = [
            cfg for cfg in setup_configs if not (configs_dir / cfg).exists()
        ]

        if missing_configs:
            print(
                f"   CONFIG: Copying {len(missing_configs)} additional config files..."
            )
            for source_configs_dir in source_configs_dirs:
                if source_configs_dir.exists():
                    for config_name in missing_configs:
                        source_config = source_configs_dir / config_name
                        if source_config.exists():
                            shutil.copy2(source_config, configs_dir / config_name)
                            print(f"         [OK] {config_name}")
                    break
            else:
                print("      WARNING: Could not find additional config templates")
                return 1

        return 0
    except Exception as e:
        print(f"   [ERROR] Config copying failed: {e}")
        return 1


def create_activation_scripts():
    """Create activation scripts for development."""
    # Main activation script
    main_script = """#!/bin/bash
# TimeFlies Development Environment

# Suppress TensorFlow/CUDA warnings and logs
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export GRPC_VERBOSITY=ERROR
export AUTOGRAPH_VERBOSITY=0
export CUDA_VISIBLE_DEVICES=""
export ABSL_LOG_LEVEL=ERROR

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    # Clean up prompt - remove any existing (.venv) and empty parentheses
    PS1="${PS1//\\(.venv\\) /}"
    PS1="${PS1//\\(.venv\\)/}"
    PS1="${PS1//\\(\\) /}"
    PS1="${PS1//\\(\\)/}"
    export PS1="(.venv) ${PS1}"
else
    echo "[ERROR] Main environment not found (.venv/bin/activate)"
    return 1
fi

# Create helpful aliases
alias tf="timeflies"
alias tf-verify="timeflies verify"
alias tf-split="timeflies split"
alias tf-train="timeflies train"
alias tf-eval="timeflies evaluate"
alias tf-test="timeflies test"

echo "DNA: TimeFlies Development Environment Activated!"
echo ""
echo "Development commands:"
echo "  timeflies test --coverage    # Run test suite with coverage"
echo "  timeflies test --fast        # Quick tests"
echo "  timeflies create-test-data   # Generate test fixtures"
echo "  timeflies verify             # System verification"
echo ""
echo "Code quality:"
echo "  ruff check src/              # Linting"
echo "  ruff format src/             # Code formatting"
"""

    # Batch activation script
    batch_script = """#!/bin/bash
# TimeFlies Batch Correction Environment

# Suppress TensorFlow/CUDA warnings and logs
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export GRPC_VERBOSITY=ERROR
export AUTOGRAPH_VERBOSITY=0
export CUDA_VISIBLE_DEVICES=""
export ABSL_LOG_LEVEL=ERROR

# Activate batch correction environment
if [[ -f ".venv_batch/bin/activate" ]]; then
    source .venv_batch/bin/activate
    # Clean up prompt - remove any existing (.venv_batch) and empty parentheses
    PS1="${PS1//\\(.venv_batch\\) /}"
    PS1="${PS1//\\(.venv_batch\\)/}"
    PS1="${PS1//\\(\\) /}"
    PS1="${PS1//\\(\\)/}"
    export PS1="(.venv_batch) ${PS1}"
else
    echo "[ERROR] Batch correction environment not found"
    return 1
fi

echo "DNA: TimeFlies Batch Correction Environment Activated!"
echo ""
echo "Available tools:"
echo "  timeflies batch-correct  # Run scVI batch correction"
echo "  python                   # PyTorch + scvi-tools available"
echo ""
echo "To return to main environment:"
echo "  deactivate"
echo "  source .activate.sh"
"""

    with open(".activate.sh", "w") as f:
        f.write(main_script)
    with open(".activate_batch.sh", "w") as f:
        f.write(batch_script)

    # Make executable

    subprocess.run(["chmod", "+x", ".activate.sh", ".activate_batch.sh"], check=True)


def tune_command(args) -> int:
    """
    Run hyperparameter tuning with grid, random, or Bayesian optimization.

    Args:
        args: Command line arguments containing tuning config path

    Returns:
        0 on success, 1 on failure
    """
    from pathlib import Path

    from common.core.hyperparameter_tuner import HyperparameterTuner

    print("\n" + "=" * 60)
    print("🔬 TIMEFLIES HYPERPARAMETER TUNING")
    print("=" * 60)
    print("Automated hyperparameter optimization for TimeFlies models")
    print("")

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Hyperparameter tuning configuration not found: {config_path}")
        return 1

    try:
        # Initialize hyperparameter tuner
        print(f"📋 Loading configuration: {config_path}")
        tuner = HyperparameterTuner(str(config_path))

        print(f"🔍 Search method: {tuner.search_method}")
        print(
            f"🎯 Number of trials: {tuner.n_trials if tuner.search_method != 'grid' else 'All combinations'}"
        )
        print("")

        # Run hyperparameter search
        resume = not args.no_resume
        results = tuner.run_search(resume=resume)

        print("\n" + "=" * 60)
        print("🎉 HYPERPARAMETER TUNING COMPLETED!")
        print("=" * 60)
        print("📊 Search Results:")
        print(f"   • Method: {results['search_method']}")
        print(f"   • Total trials: {results['total_trials']}")
        print(f"   • Completed: {results['completed_trials']}")
        print(f"   • Failed: {results['failed_trials']}")
        print(f"   • Duration: {results['total_time'] / 60:.1f} minutes")
        print("")

        if results["best_trial"]:
            best = results["best_trial"]
            print("🏆 Best Configuration:")
            print(f"   • Variant: {best['variant_name']}")
            print(f"   • Model: {best['model_type']}")
            if best["metrics"]:
                print(f"   • Accuracy: {best['metrics'].get('accuracy', 'N/A'):.4f}")
                print(f"   • F1-Score: {best['metrics'].get('f1_score', 'N/A'):.4f}")
            print(f"   • Parameters: {best['hyperparameters']}")
            print("")

        print("📁 Results Location:")
        print(f"   • Report: {results['report_path']}")
        print(f"   • Metrics: {results['metrics_path']}")
        print(f"   • Directory: {results['output_directory']}")
        print("")

        # Additional Bayesian optimization info
        if results.get("optuna_study"):
            study = results["optuna_study"]
            print("🧠 Bayesian Optimization Details:")
            print(f"   • Study: {study['study_name']}")
            print(f"   • Best value: {study['best_value']:.4f}")
            print("")

        print("💡 Next Steps:")
        print("   1. Review detailed report for comprehensive analysis")
        print("   2. Use best configuration for production training")
        print("   3. Consider re-running with different search method")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"[ERROR] Hyperparameter tuning failed: {e}")
        return 1


def queue_command(args) -> int:
    """
    Run automated model queue for sequential training.

    Args:
        args: Command line arguments containing queue config path

    Returns:
        0 on success, 1 on failure
    """
    from pathlib import Path

    from common.core.model_queue import ModelQueueManager

    print("\n" + "=" * 60)
    print("TIMEFLIES MODEL QUEUE MANAGER")
    print("=" * 60)

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"[ERROR] Queue configuration not found: {config_path}")
        print("\nExample queue configuration: configs/model_queue_example.yaml")
        return 1

    try:
        # Initialize queue manager
        manager = ModelQueueManager(str(config_path))

        # Run the queue (resume by default unless --no-resume is specified)
        resume = not args.no_resume
        manager.run_queue(resume=resume)

        return 0

    except Exception as e:
        print(f"[ERROR] Queue execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def update_command(args) -> int:
    """Update TimeFlies to the latest version from GitHub."""
    import sys
    from pathlib import Path

    print("PROCESS: Updating TimeFlies to latest version...")
    print("=" * 50)
    print("POLICY: System files will be updated, user data preserved:")
    print("   • UPDATE: .timeflies_src/, TimeFlies_Launcher.py, official templates")
    print("   • PRESERVE: data/, outputs/, configs/, custom templates")
    print("=" * 50)

    try:
        # Check if git is available
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "[ERROR] Git not found. Please install git to use the update command."
            )
            return 1

        # Create temporary directory for update
        import tempfile

        with tempfile.TemporaryDirectory(prefix="timeflies_update_") as temp_dir:
            temp_path = Path(temp_dir)

            print("LOADING: Downloading latest version...")

            # Clone the latest version
            repo_url = "git@github.com:rsinghlab/TimeFlies.git"
            clone_result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "-b",
                    "main",
                    repo_url,
                    str(temp_path / "timeflies_update"),
                ],
                capture_output=True,
                text=True,
            )

            if clone_result.returncode != 0:
                print(f"[ERROR] Failed to download update: {clone_result.stderr}")
                print("NOTE: Please check your GitHub access and try again")
                return 1

            print("PACKAGE: Installing updated version...")

            # Install the updated version
            update_path = temp_path / "timeflies_update"
            install_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", str(update_path)],
                capture_output=True,
                text=True,
            )

            if install_result.returncode != 0:
                print(f"[ERROR] Installation failed: {install_result.stderr}")
                return 1

            print("[OK] TimeFlies updated successfully!")

            # Smart update: refresh system files, preserve user data
            print("UPDATE: Updating system files while preserving user data...")
            import shutil

            # 1. Update .timeflies_src (source code and templates)
            print("   SYSTEM: Updating .timeflies_src with latest code...")
            timeflies_src = Path.cwd() / ".timeflies_src"
            if timeflies_src.exists():
                shutil.rmtree(timeflies_src)
            shutil.copytree(update_path, timeflies_src)
            print("      [OK] Source code updated")

            # 2. Update GUI launcher (preserve user customizations by checking if different)
            launcher_file = Path("TimeFlies_Launcher.py")
            new_launcher = update_path / "TimeFlies_Launcher.py"
            if new_launcher.exists():
                if not launcher_file.exists():
                    print("   GUI: Installing TimeFlies_Launcher.py...")
                    shutil.copy2(new_launcher, launcher_file)
                    print("      [OK] GUI launcher installed")
                else:
                    # Compare files and only update if different
                    import filecmp

                    if not filecmp.cmp(new_launcher, launcher_file, shallow=False):
                        print("   GUI: Updating TimeFlies_Launcher.py...")
                        shutil.copy2(new_launcher, launcher_file)
                        print("      [OK] GUI launcher updated")
                    else:
                        print("   GUI: TimeFlies_Launcher.py already current")

            # 3. Update templates (add new, preserve custom ones)
            print("   TEMPLATES: Updating analysis templates...")
            templates_dir = Path("templates")
            new_templates_dir = update_path / "templates"
            if new_templates_dir.exists():
                if not templates_dir.exists():
                    templates_dir.mkdir(parents=True, exist_ok=True)

                # Copy new/updated template files
                for template_file in new_templates_dir.glob("*"):
                    dst_template = templates_dir / template_file.name
                    if not dst_template.exists():
                        shutil.copy2(template_file, dst_template)
                        print(f"      [NEW] {template_file.name}")
                    elif template_file.name in [
                        "README.md",
                        "custom_analysis_example.py",
                        "aging_analysis_template.py",
                    ]:
                        # Update official templates
                        shutil.copy2(template_file, dst_template)
                        print(f"      [UPDATED] {template_file.name}")
                    # Preserve user's custom templates (don't overwrite)

            # 4. Add any missing config files (preserve existing ones)
            print("   CONFIG: Checking for missing configuration files...")
            config_copy_result = setup_user_environment()
            if config_copy_result == 0:
                print("      [OK] Configuration files current")
            else:
                print("      WARNING: Some config files may not have been updated")

            print("   [OK] System files updated, user data preserved")

            # Test the updated installation
            print("TESTING: Testing updated installation...")
            test_result = subprocess.run(
                ["timeflies", "--help"], capture_output=True, text=True
            )

            if test_result.returncode == 0:
                print("[OK] Update completed successfully!")
                print("\nSUCCESS: TimeFlies is now up to date!")
            else:
                print("WARNING:  Update installed but CLI test failed")
                print(
                    "NOTE: You may need to restart your terminal or reactivate your environment"
                )

            return 0

    except Exception as e:
        print(f"[ERROR] Update failed: {e}")
        return 1
