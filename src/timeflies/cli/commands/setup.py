"""
TimeFlies CLI Setup Commands

Contains setup, split, and environment configuration commands.
"""

import os
from pathlib import Path


def new_setup_command(args) -> int:
    """
    Complete setup workflow for users: split data + optional batch correction + verify system.
    For developers: just create environments.
    """
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
        from .training import batch_command

        batch_result = batch_command(args)
        if batch_result != 0:
            print("[ERROR] Batch correction failed.")
            return batch_result
        print("[OK] Batch correction completed")
    else:
        print("\n2. Skipping batch correction")

    # 3. Create output directories
    print("\n3. Creating output directories...")

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
    from timeflies.cli.system_checks import verify_system

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
        from timeflies.data.setup import DataSetupManager

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


def setup_user_environment(quiet_mode=False):
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
            if not quiet_mode:
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
            if not quiet_mode:
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
            if not quiet_mode:
                print("   DOC: templates/ directory already exists")

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
