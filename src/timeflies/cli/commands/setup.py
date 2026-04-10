"""
TimeFlies CLI Setup Commands

Contains setup, split, and environment configuration commands.
"""

import os
from pathlib import Path


def new_setup_command(args) -> int:
    """Complete setup: create directories, split data, optional batch correction, verify."""
    print("LAUNCH: TimeFlies Complete Setup")
    print("=" * 50)

    # 0. Create user directories
    print("\n0. Setting up directories...")
    setup_result = setup_user_environment()
    if setup_result != 0:
        return setup_result
    print("[OK] Directories ready")

    # 1. Create data splits
    print("\n1. Creating data splits...")
    if hasattr(args, "force_split") and args.force_split:
        print("   FORCE: Will recreate existing splits")
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
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("[OK] Output directories created")

    # 4. Verify
    print("\n4. Verifying system setup...")
    from timeflies.cli.system_checks import verify_system

    dev_mode = hasattr(args, "dev") and args.dev
    if not verify_system(dev_mode=dev_mode):
        print("[ERROR] System verification failed.")
        return 1

    print("\nSUCCESS: SETUP COMPLETE!")
    print("=" * 50)
    print("Next steps:")
    print("  timeflies eda --save-report")
    print("  timeflies train")
    print("  timeflies evaluate")
    print("  timeflies analyze")
    return 0


def split_command(args) -> int:
    """Create train/eval data splits from original data."""
    try:
        from timeflies.data.setup import DataSetupManager

        print("PROCESS: Creating train/eval data splits...")

        force_split = hasattr(args, "force_split") and args.force_split
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


def setup_user_environment(quiet_mode: bool = False) -> int:
    """Create configs/ and examples/ directories if missing."""
    try:
        Path("configs").mkdir(parents=True, exist_ok=True)
        Path("examples").mkdir(parents=True, exist_ok=True)
        return 0
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        return 1
