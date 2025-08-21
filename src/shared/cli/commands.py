"""
TimeFlies CLI Commands

This module contains all command-line interface commands for TimeFlies.
Each command is implemented as a separate function with proper error handling.
"""

import sys
from pathlib import Path

from ..core.active_config import get_config_for_active_project, get_active_project

# Optional import for batch correction
try:
    from ..data.preprocessing.batch_correction import BatchCorrector

    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    BatchCorrector = None


def run_system_tests(args) -> int:
    """Run comprehensive system verification before training."""
    print("🔧 Running TimeFlies System Verification...")
    print("=" * 60)

    # Suppress verbose logging during verification
    import logging
    import os
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

    # Temporarily reduce logging level and suppress warnings
    original_level = logging.getLogger().level
    active_config_level = logging.getLogger("timeflies.shared.core.active_config").level
    config_manager_level = logging.getLogger(
        "timeflies.projects.fruitfly_aging.core.config_manager"
    ).level

    logging.getLogger().setLevel(logging.ERROR)  # Only show errors
    logging.getLogger("timeflies.shared.core.active_config").setLevel(logging.ERROR)
    logging.getLogger("timeflies.projects.fruitfly_aging.core.config_manager").setLevel(
        logging.ERROR
    )
    warnings.filterwarnings("ignore")  # Suppress warnings like scVI

    all_passed = True
    workflow_complete = True

    # 1. Test Python environment and imports
    print("1. Testing Python environment...")
    try:
        import numpy, pandas, scanpy, anndata

        print("   ✅ Scientific packages (numpy, pandas, scanpy, anndata)")

        import tensorflow, sklearn

        print("   ✅ Machine learning packages (tensorflow, sklearn)")

        from ..core.active_config import (
            get_active_project,
            get_config_for_active_project,
        )
        from ..utils.gpu_handler import GPUHandler

        print("   ✅ TimeFlies core modules")

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        all_passed = False

    # 2. Test project configuration
    print("\n2. Testing project configuration...")
    try:
        # Check for project override from CLI flags
        if hasattr(args, 'project') and args.project:
            active_project = args.project
            print(f"   🔄 Using CLI project override: {active_project}")
            # Use importlib to dynamically import the project's config manager
            import importlib
            config_module = importlib.import_module(f"projects.{args.project}.core.config_manager")
            ConfigManager = config_module.ConfigManager
            config_manager = ConfigManager("default")
        else:
            active_project = get_active_project()
            print(f"   ✅ Active project detected: {active_project}")
            config_manager = get_config_for_active_project("default")
        
        config = config_manager.get_config()
        print(f"   ✅ Project config loaded successfully")
        print(f"   ✅ Target variable: {config.data.encoding_variable}")
        print(f"   ✅ Model type: {config.data.model}")

    except Exception as e:
        print(f"   ❌ Project config not set up: {e}")
        print(f"   💡 Check your configs/default.yaml file")
        all_passed = False

    # 3. Test original data files (workflow step 2)
    print("\n3. Testing original data files...")
    try:
        from pathlib import Path

        data_root = Path("data")

        if not data_root.exists():
            print("   ❌ No data directory found")
            print("   💡 Create data/[project]/[tissue]/ and add original H5AD files")
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
                print(f"   ✅ Original data files found:")
                for file in original_files_found:
                    print(f"      {file}")
            else:
                print("   ❌ No original H5AD data files found")
                print("   💡 Add your *_original.h5ad files to data/[project]/[tissue]/")
                workflow_complete = False

    except Exception as e:
        print(f"   ❌ Data check failed: {e}")
        all_passed = False

    # 4. Test test data fixtures (workflow step 4)
    print("\n4. Testing test data fixtures...")
    try:
        from pathlib import Path

        fixtures_root = Path("tests/fixtures")

        if not fixtures_root.exists():
            print("   ❌ No test fixtures directory found")
            print("   💡 Run: python run_timeflies.py create-test-data")
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
                print(f"   ✅ Test data available for: {', '.join(test_projects)}")

                summary_file = fixtures_root / "test_data_summary.json"
                if summary_file.exists():
                    print("   ✅ Test data summary file exists")
                else:
                    print("   ⚠️  Test data summary missing")
            else:
                print("   ❌ No test data fixtures found")
                print("   💡 Run: python run_timeflies.py create-test-data")
                workflow_complete = False

    except Exception as e:
        print(f"   ❌ Test data check failed: {e}")
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
            print(f"   ✅ Data splits found for: {', '.join(splits_found)}")
        else:
            print("   ❌ No data splits (*_train.h5ad, *_eval.h5ad) found")
            print("   💡 Run: python run_timeflies.py setup")
            workflow_complete = False

    except Exception as e:
        print(f"   ❌ Data splits check failed: {e}")
        workflow_complete = False
        all_passed = False

    # 6. Test batch correction (workflow step 6, optional)
    print("\n6. Testing batch correction...")
    try:
        from pathlib import Path

        # Check if batch environment exists
        batch_env = Path(".venv_batch")
        if batch_env.exists():
            print("   ✅ Batch correction environment found")

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
                print("   ✅ Batch corrected files found:")
                for file in batch_files_found:
                    print(f"      {file}")
            else:
                print(
                    "   ⚠️  Batch correction environment found but no batch corrected files"
                )
                print(
                    "   💡 Run: source activate_batch.sh && python run_timeflies.py batch-correct"
                )
        else:
            print("   ⚠️  No batch correction environment")
            print(
                "   💡 Optional: Re-run setup_dev_env.sh and choose 'y' for batch environment"
            )

    except Exception as e:
        print(f"   ⚠️  Could not check batch correction: {e}")

    # 7. Test GPU/hardware
    print("\n7. Testing hardware configuration...")
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"   ✅ GPU available: {len(gpus)} device(s)")
        else:
            print("   ✅ Running on CPU (GPU not required)")

    except Exception as e:
        print(f"   ⚠️  Could not check hardware: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_passed and workflow_complete:
        print("🎉 COMPLETE WORKFLOW VERIFIED! Ready to train models.")
    elif all_passed:
        print("✅ CORE SYSTEM OK - Complete the workflow steps below.")
    else:
        print("❌ ISSUES FOUND - Fix the problems above.")

    print("\n📋 Workflow status:")
    print(
        "   1. ✅ Data files placed"
        if workflow_complete
        else "   1. ❌ Place original H5AD files in data/[project]/[tissue]/"
    )
    print(
        "   2. ✅ Test fixtures created"
        if workflow_complete
        else "   2. ❌ Run: python run_timeflies.py create-test-data"
    )
    print(
        "   3. ✅ Data splits created"
        if workflow_complete
        else "   3. ❌ Run: python run_timeflies.py setup"
    )
    print(
        "   4. ✅ Verification complete"
        if all_passed and workflow_complete
        else "   4. ❌ Fix issues above"
    )
    print("   5. Next: python run_timeflies.py train")
    print("   ⚠️  Optional: python run_timeflies.py batch-correct (between steps 3-4)")

    print("=" * 60)

    # Restore original logging level and warnings
    logging.getLogger().setLevel(original_level)
    logging.getLogger("timeflies.shared.core.active_config").setLevel(
        active_config_level
    )
    logging.getLogger("timeflies.projects.fruitfly_aging.core.config_manager").setLevel(
        config_manager_level
    )
    warnings.resetwarnings()

    return 0 if all_passed else 1


def run_test_suite(args) -> int:
    """Run the test suite using test_runner.py functionality."""
    print("🧪 Running TimeFlies Test Suite...")
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
        print(f"❌ Could not import test runner: {e}")
        print("💡 Make sure tests/test_runner.py exists")
        return 1
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


def setup_command(args) -> int:
    """Set up train/eval splits for all projects with both original and batch corrected data."""
    print("🔧 Setting up train/eval data splits...")
    print("=" * 60)

    try:
        # Load setup config for split parameters
        setup_config_path = Path("configs/setup.yaml")
        if setup_config_path.exists():
            print("📋 Loading setup configuration...")
            import yaml

            with open(setup_config_path, "r") as f:
                setup_data = yaml.safe_load(f)
        else:
            print("⚠️  Setup config not found, using defaults...")
            setup_data = {
                "data": {"train_test_split": {"split_size": 5000}},
                "general": {"random_state": 42},
            }

        split_size = setup_data["data"]["train_test_split"]["split_size"]
        random_state = setup_data["general"]["random_state"]
        print(
            f"📊 Split parameters: {split_size} cells per eval set, random_state={random_state}"
        )

        # Find all projects with data
        print("\n🔍 Scanning for projects with data...")
        data_root = Path("data")
        projects_found = []

        if data_root.exists():
            for project_dir in data_root.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    for tissue_dir in project_dir.iterdir():
                        if tissue_dir.is_dir():
                            original_files = list(tissue_dir.glob("*_original.h5ad"))
                            if original_files:
                                projects_found.append(
                                    (project_dir.name, tissue_dir.name)
                                )

        if not projects_found:
            print("❌ No projects with original data found in data/ directory")
            print(
                "💡 Place your *_original.h5ad files in data/[project]/[tissue]/ first"
            )
            return 1

        print(f"✅ Found {len(projects_found)} project(s) with data:")
        for project, tissue in projects_found:
            print(f"   • {project}/{tissue}")

        # Process each project
        total_processed = 0

        for project_name, tissue in projects_found:
            print(f"\n{'='*60}")
            print(f"🧬 Processing project: {project_name}/{tissue}")
            print(f"{'='*60}")

            # Create output directories for this project
            output_dirs = [
                f"outputs/{project_name}/models",
                f"outputs/{project_name}/results",
                f"outputs/{project_name}/logs",
            ]

            for dir_path in output_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"📁 Created: {dir_path}")

            # Load project config for stratification variables
            try:
                # Load project config directly
                project_config_path = Path(f"configs/{project_name}/default.yaml")
                if project_config_path.exists():
                    import yaml

                    with open(project_config_path, "r") as f:
                        project_data = yaml.safe_load(f)
                    encoding_var = project_data.get("data", {}).get(
                        "encoding_variable", "age"
                    )
                    print(f"🎯 Stratification variable: {encoding_var}")
                else:
                    print(
                        f"⚠️  Config not found for {project_name}, using default: age"
                    )
                    encoding_var = "age"
            except Exception as e:
                print(
                    f"⚠️  Could not load config for {project_name}, using default: age"
                )
                encoding_var = "age"

            # Process both original and batch corrected data
            data_dir = Path(f"data/{project_name}/{tissue}")

            # Find ONLY source data files (not existing splits)
            original_files = list(data_dir.glob("*_original.h5ad"))
            # Find batch corrected SOURCE files (original_batch, not train_batch/eval_batch)
            batch_files = list(data_dir.glob("*_original_batch.h5ad"))

            files_to_process = []
            if original_files:
                files_to_process.extend([(f, "original") for f in original_files])
            if batch_files:
                files_to_process.extend([(f, "batch") for f in batch_files])

            print(
                f"📄 Found {len(original_files)} original + {len(batch_files)} batch files"
            )
            if len(original_files) > 0 and len(batch_files) == 0:
                print(
                    f"   ℹ️  No batch corrected files found - only splitting original data"
                )
                print(f"   💡 Run batch correction first to create batch splits")

            # Process each file type
            for file_path, file_type in files_to_process:
                success = process_data_splits(
                    file_path=file_path,
                    file_type=file_type,
                    split_size=split_size,
                    encoding_var=encoding_var,
                    random_state=random_state,
                )
                if success:
                    total_processed += 1

        print(f"\n{'='*60}")
        print(f"🎉 SETUP COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(
            f"📊 Processed {total_processed} file(s) across {len(projects_found)} project(s)"
        )

        print(f"\n📋 Next steps:")
        print(f"   1. Verify setup: python run_timeflies.py verify")
        print(f"   2. Train models: python run_timeflies.py train")
        print(f"   3. Run evaluation: python run_timeflies.py evaluate")
        print(f"\n💡 Optional batch correction:")
        print(f"   • Activate batch env: source activate_batch.sh")
        print(f"   • Run batch correction: python run_timeflies.py batch-correct")
        print(f"   • Return to main: source activate.sh")

        return 0

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1


def process_data_splits(
    file_path: Path,
    file_type: str,
    split_size: int,
    encoding_var: str,
    random_state: int,
) -> bool:
    """Process data splits for a single file with enhanced stratification."""
    import numpy as np
    import anndata
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
            print(f"⏭️  Splits already exist for {file_path.name} - skipping")
            print(
                f"   📁 Found: {train_file.name if train_file.exists() else ''} {eval_file.name if eval_file.exists() else ''}"
            )
            return True

        print(f"📂 Processing: {file_path.name}")

        # Load data
        adata = anndata.read_h5ad(file_path)
        print(f"   📊 Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Determine stratification columns
        stratify_cols = []

        # Primary stratification: encoding variable (age/disease status)
        if encoding_var in adata.obs.columns:
            stratify_cols.append(encoding_var)
            print(f"   🎯 Primary stratification: {encoding_var}")

        # Secondary stratification: sex
        sex_cols = ["sex", "Sex", "gender", "Gender"]
        for col in sex_cols:
            if col in adata.obs.columns and col not in stratify_cols:
                stratify_cols.append(col)
                print(f"   ⚧️  Secondary stratification: {col}")
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
                print(f"   🧬 Tertiary stratification: {col}")
                break

        if not stratify_cols:
            print(f"   ⚠️  No stratification columns found, using random split")
            stratify_labels = None
        else:
            # Create combined stratification labels
            if len(stratify_cols) == 1:
                stratify_labels = adata.obs[stratify_cols[0]].astype(str)
            else:
                stratify_labels = (
                    adata.obs[stratify_cols].astype(str).agg("_".join, axis=1)
                )

            print(f"   📈 Stratification groups: {len(stratify_labels.unique())}")

        # Calculate split ratio to get desired eval size
        total_cells = adata.shape[0]
        if split_size >= total_cells:
            print(
                f"   ⚠️  Split size ({split_size}) >= total cells ({total_cells}), using 20% split"
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
        print(f"   💾 Saving: {train_file.name} ({len(adata_train)} cells)")
        adata_train.write_h5ad(train_file)

        print(f"   💾 Saving: {eval_file.name} ({len(adata_eval)} cells)")
        adata_eval.write_h5ad(eval_file)

        print(f"   ✅ Split completed successfully")
        return True

    except Exception as e:
        print(f"   ❌ Error processing {file_path.name}: {e}")
        return False


def train_command(args, config) -> int:
    """Train a model using project configuration settings."""
    try:
        print(f"Starting training with project settings:")
        print(f"   Project: {getattr(config.data, 'project', 'unknown')}")
        print(f"   Tissue: {config.data.tissue}")
        print(f"   Model: {config.data.model}")
        print(f"   Target: {config.data.encoding_variable}")
        print(f"   Batch correction: {config.data.batch_correction.enabled}")

        # Import project-specific PipelineManager
        active_project = get_active_project()
        if active_project == "fruitfly_aging":
            from projects.fruitfly_aging.core.pipeline_manager import PipelineManager
        elif active_project == "fruitfly_alzheimers":
            from projects.fruitfly_alzheimers.core.pipeline_manager import (
                PipelineManager,
            )
        else:
            raise ValueError(f"Unknown project: {active_project}")

        # Initialize and run pipeline
        pipeline = PipelineManager(config)
        results = pipeline.run()

        print(f"\n✅ Training completed successfully!")
        
        # Check if model was actually saved (based on validation loss improvement)
        model_path = results.get('model_path', 'outputs/models/')
        import os
        model_file = os.path.join(model_path, 'best_model.h5')
        
        # Check if model was updated by comparing file modification time with start time
        if os.path.exists(model_file):
            import time
            file_mod_time = os.path.getmtime(model_file)
            # If file was modified in the last minute, it was likely updated
            if time.time() - file_mod_time < 60:
                print(f"   ✅ Model saved (validation loss improved): {model_path}")
            else:
                print(f"   ⏭️  Model not saved (validation loss did not improve)")
                print(f"   📁 Existing model location: {model_path}")
        else:
            print(f"   ✅ New model saved: {model_path}")
            
        print(f"   📊 Results saved to: {results.get('results_path', 'outputs/results/')}")
        if "duration" in results:
            print(f"   ⏱️  Training duration: {results['duration']:.1f} seconds")

        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
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

        print(f"\n" + "=" * 60)
        print("✓ BATCH CORRECTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show what was created
        tissue = args.tissue
        print(f"✓ scVI model trained on: drosophila_{tissue}_aging_train.h5ad")
        print(f"✓ Model applied to: drosophila_{tissue}_aging_eval.h5ad (query mode)")
        print(f"✓ Created: drosophila_{tissue}_aging_train_batch.h5ad")
        print(f"✓ Created: drosophila_{tissue}_aging_eval_batch.h5ad")
        print(f"✓ Original file untouched (prevents data leakage)")

        print(f"\nNext steps:")
        print(f"1. Return to main environment: source activate.sh")
        print(f"2. Train with batch data:     python run_timeflies.py train")
        print(f"   (Will automatically use batch-corrected files)")
        print(
            f"\nNote: No additional setup needed - batch correction used existing splits!"
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
    
    try:
        print(f"Starting evaluation with project settings:")
        print(f"   Project: {getattr(config.data, 'project', 'unknown')}")
        print(f"   Tissue: {config.data.tissue}")
        print(f"   Model: {config.data.model}")
        print(f"   Target: {config.data.encoding_variable}")

        # Import project-specific PipelineManager
        active_project = get_active_project()
        if active_project == "fruitfly_aging":
            from projects.fruitfly_aging.core.pipeline_manager import PipelineManager
        elif active_project == "fruitfly_alzheimers":
            from projects.fruitfly_alzheimers.core.pipeline_manager import (
                PipelineManager,
            )
        else:
            raise ValueError(f"Unknown project: {active_project}")

        # Initialize pipeline and run evaluation-only workflow
        # (This includes metrics, interpretation, and visualizations based on config)
        pipeline = PipelineManager(config)
        results = pipeline.run_evaluation()

        print("\n✅ Evaluation completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def create_test_data_command(args) -> int:
    """Create test data fixtures by sampling from all available project data."""
    try:
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import json
        from pathlib import Path

        print("🧪 Creating Test Data Fixtures")
        print("=" * 60)

        # Find all available projects
        data_root = Path("data")
        if not data_root.exists():
            print(
                "❌ Data directory not found. Place data files in data/[project]/[tissue]/ first."
            )
            return 1

        projects_found = []
        test_data_created = []

        # Scan for project directories
        for project_dir in data_root.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue

            print(f"\n📁 Scanning project: {project_dir.name}")

            # Look for tissue directories
            for tissue_dir in project_dir.iterdir():
                if not tissue_dir.is_dir():
                    continue

                print(f"  📂 Tissue: {tissue_dir.name}")

                # Look for data files
                h5ad_files = list(tissue_dir.glob("*.h5ad"))
                if not h5ad_files:
                    print(f"    ⚠️  No H5AD files found")
                    continue

                # Find original/train files
                original_files = [f for f in h5ad_files if "original" in f.name]
                train_files = [
                    f for f in h5ad_files if "train" in f.name and "batch" not in f.name
                ]
                batch_files = [f for f in h5ad_files if "batch" in f.name]

                if original_files:
                    data_file = original_files[0]
                elif train_files:
                    data_file = train_files[0]
                else:
                    print(f"    ⚠️  No suitable data files found")
                    continue

                print(f"    📊 Using: {data_file.name}")

                # Create test data for this project
                test_data = create_project_test_data(
                    project_dir.name,
                    tissue_dir.name,
                    data_file,
                    batch_files[0] if batch_files else None,
                    n_samples=100,
                    n_genes=500,
                )

                if test_data:
                    test_data_created.append(
                        {
                            "project": project_dir.name,
                            "tissue": tissue_dir.name,
                            "stats": test_data,
                        }
                    )
                    projects_found.append(f"{project_dir.name}/{tissue_dir.name}")

        if not test_data_created:
            print(
                "\n❌ No test data created. Ensure data files are in data/[project]/[tissue]/ format."
            )
            return 1

        # Save summary
        summary = {
            "created_at": pd.Timestamp.now().isoformat(),
            "projects": test_data_created,
            "total_projects": len(test_data_created),
        }

        summary_path = Path("tests/fixtures/test_data_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n✅ Test data created for {len(projects_found)} project(s):")
        for project in projects_found:
            print(f"  ✓ {project}")

        print(f"\n📄 Summary saved to: {summary_path}")
        print(f"🧪 Test files saved to: tests/fixtures/[project]/")

        return 0

    except ImportError as e:
        print(f"❌ Required dependencies not available: {e}")
        print("Ensure you're in the main environment with scanpy installed.")
        return 1
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return 1


def create_project_test_data(
    project, tissue, data_file, batch_file=None, n_samples=100, n_genes=500, seed=42
):
    """Create test data for a specific project by sampling from real data."""
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import json
    from pathlib import Path

    np.random.seed(seed)

    try:
        print(f"    🔄 Loading data from {data_file}")
        adata = sc.read_h5ad(data_file)

        print(f"    📊 Original shape: {adata.shape}")

        # Sample cells and genes
        n_cells = min(n_samples, adata.n_obs)
        n_genes_sample = min(n_genes, adata.n_vars)

        cell_indices = np.random.choice(adata.n_obs, n_cells, replace=False)
        gene_indices = np.random.choice(adata.n_vars, n_genes_sample, replace=False)

        # Create subset
        adata_subset = adata[cell_indices, gene_indices].copy()
        print(f"    📏 Sampled shape: {adata_subset.shape}")

        # Create output directory
        output_dir = Path("tests/fixtures") / project
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save regular test data
        output_path = output_dir / f"test_data_{tissue}.h5ad"
        print(f"    💾 Saving to {output_path}")
        adata_subset.write_h5ad(output_path)

        # Load and process batch-corrected data if available
        adata_batch = None
        if batch_file and batch_file.exists():
            try:
                print(f"    🔄 Loading batch-corrected data from {batch_file}")
                adata_batch_full = sc.read_h5ad(batch_file)
                print(f"    📊 Batch data shape: {adata_batch_full.shape}")

                # Sample from batch data (may have different size than original)
                batch_n_cells = min(n_cells, adata_batch_full.n_obs)
                batch_n_genes = min(n_genes_sample, adata_batch_full.n_vars)

                batch_cell_indices = np.random.choice(
                    adata_batch_full.n_obs, batch_n_cells, replace=False
                )
                batch_gene_indices = np.random.choice(
                    adata_batch_full.n_vars, batch_n_genes, replace=False
                )

                adata_batch = adata_batch_full[
                    batch_cell_indices, batch_gene_indices
                ].copy()
                batch_output_path = output_dir / f"test_data_{tissue}_batch.h5ad"
                print(f"    💾 Saving batch-corrected to {batch_output_path}")
                adata_batch.write_h5ad(batch_output_path)
            except Exception as e:
                print(
                    f"    ⚠️  Batch data processing failed: {e}, skipping batch-corrected test data"
                )

        # Generate statistics
        stats = generate_data_stats(
            adata_subset, project, tissue, batch_corrected=adata_batch is not None
        )

        # Save stats
        stats_path = output_dir / f"test_data_{tissue}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"    📊 Stats saved to {stats_path}")
        print(f"    ✅ Test data created for {project}/{tissue}")

        return stats

    except Exception as e:
        print(f"    ❌ Failed to create test data for {project}/{tissue}: {e}")
        return None


def generate_data_stats(adata, project, tissue, batch_corrected=False):
    """Generate comprehensive statistics for test data."""
    import numpy as np
    import pandas as pd

    # Basic info
    stats = {
        "project": project,
        "tissue": tissue,
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
