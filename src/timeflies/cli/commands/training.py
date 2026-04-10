"""
TimeFlies CLI Training Commands

Contains training, evaluation, and batch correction commands.
"""

import os
from pathlib import Path

from ._utils import BatchCorrector, suppress_stderr


def train_command(args, config) -> int:
    """Train a model using project configuration settings."""
    # Run EDA first if requested
    if hasattr(args, "with_eda") and args.with_eda:
        from .analysis import eda_command

        print("\nDATA: Running EDA before training...")
        result = eda_command(args, config)
        if result != 0:
            print("[ERROR] EDA failed, stopping pipeline")
            return result

    # Get experiment details for cleaner display
    target = getattr(config.data, "target_variable", "unknown")
    tissue = getattr(config.data, "tissue", "unknown")
    model_type = getattr(config.data, "model", "unknown")

    # Use common PipelineManager for all projects (GPU will be configured there)
    with suppress_stderr():
        with suppress_stderr():
            from timeflies.core import PipelineManager

    # Generate proper experiment name with sex/cell type filters
    from ...utils.split_naming import SplitNamingUtils

    experiment_suffix = SplitNamingUtils.generate_experiment_suffix(config)
    display_format = (
        f"{tissue.lower()}_{model_type.lower()}_{target.lower()}_{experiment_suffix}"
    )

    # Initialize and run pipeline in training mode
    from ...display.display_manager import DisplayManager

    pipeline = PipelineManager(config, mode="training")
    display_manager = DisplayManager(config)
    display_manager.print_project_and_dataset_overview(
        config, pipeline, display_format, mode="training"
    )
    results = pipeline.run_pipeline()

    # Training completed message moved to duration display

    # Run analysis after training if requested
    if hasattr(args, "with_analysis") and args.with_analysis:
        from .analysis import analyze_command

        print("\nRESEARCH: Running analysis after training...")
        result = analyze_command(args, config)
        if result != 0:
            print("WARNING:  Analysis failed but training was successful")

    # Check if model was actually saved (based on validation loss improvement)
    model_path = results.get("model_path", "outputs/models/")

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
        pass  # Model status will be shown in completion summary

    return 0


def batch_command(args) -> int:
    """Run batch correction pipeline."""

    print(f"Running batch correction for {args.tissue} tissue...")

    # Determine project from CLI args or config
    from ...core.active_config import get_config_for_active_project

    if hasattr(args, "project") and args.project:
        project = args.project
    else:
        try:
            config = get_config_for_active_project()
            project = getattr(config.data, "project", "fruitfly_aging")
        except Exception:
            project = "fruitfly_aging"

    base_dir = Path(f"data/{project}/{args.tissue}")

    # Check if batch correction is enabled for this project
    try:
        from pathlib import Path as ConfigPath

        import yaml

        # Load batch correction config to check if enabled
        config_paths = [
            ConfigPath("configs/batch_correction.yaml"),
            ConfigPath(__file__).parent.parent.parent.parent
            / "configs/batch_correction.yaml",
        ]

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break

        if config_path:
            with open(config_path) as f:
                batch_config = yaml.safe_load(f)

            # Check project-specific enabled setting, fall back to default
            enabled = batch_config.get("batch_correction", {}).get(
                "enabled", True
            )  # Default enabled
            if (
                "project_overrides" in batch_config
                and project in batch_config["project_overrides"]
            ):
                project_config = batch_config["project_overrides"][project]
                if "batch_correction" in project_config:
                    enabled = project_config["batch_correction"].get("enabled", enabled)

            if not enabled:
                print(f"❌ Batch correction is disabled for project '{project}'")
                print(
                    f"   To enable: Set 'enabled: true' in configs/batch_correction.yaml under project_overrides.{project}"
                )
                return 0

            print(f"✅ Batch correction is enabled for project '{project}'")
        else:
            print("⚠️  Could not find batch_correction.yaml, assuming enabled")

    except Exception as e:
        print(f"⚠️  Could not check enabled status: {e}, assuming enabled")

    try:
        # Try to instantiate BatchCorrector (will check dependencies)
        batch_corrector = BatchCorrector(
            tissue=args.tissue,
            base_dir=base_dir,
            project=project,
        )
        print("DEBUG: BatchCorrector created successfully")

    except ImportError:
        print("Batch correction dependencies not installed.")
        print("Install with: uv pip install 'timeflies[batch-correction]'")
        return 1

    # Dependencies available, run batch correction
    try:
        print("Training scVI model on TRAINING data only...")
        print("Then applying trained model to evaluation data (query mode)...")
        print("This prevents data leakage into the holdout set.")
        batch_corrector.run_batch_correction()

        print(f"\nDone. Created batch-corrected files in {base_dir}/")
        print("\nNext steps:")
        print("  timeflies --batch-corrected train")
        return 0

    except Exception as e:
        print(f"Batch correction failed: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def evaluate_command(args, config) -> int:
    """Evaluate a trained model using project configuration settings."""
    try:
        # Run EDA first if requested
        if hasattr(args, "with_eda") and args.with_eda:
            from .analysis import eda_command

            print("\nDATA: Running EDA before evaluation...")
            result = eda_command(args, config)
            if result != 0:
                print("[ERROR] EDA failed, stopping pipeline")
                return result

        # Get experiment details for cleaner display
        target = getattr(config.data, "target_variable", "unknown")
        tissue = getattr(config.data, "tissue", "unknown")
        model_type = getattr(config.data, "model", "unknown")

        # Use common PipelineManager for all projects (GPU will be configured there)
        with suppress_stderr():
            with suppress_stderr():
                from timeflies.core import PipelineManager

        # Generate proper experiment name with sex/cell type filters
        from ...utils.split_naming import SplitNamingUtils

        experiment_suffix = SplitNamingUtils.generate_experiment_suffix(config)
        display_format = f"{tissue.lower()}_{model_type.lower()}_{target.lower()}_{experiment_suffix}"

        # Initialize and run pipeline in evaluation mode
        from ...display.display_manager import DisplayManager

        pipeline = PipelineManager(config, mode="evaluation")
        display_manager = DisplayManager(config)
        display_manager.print_project_and_dataset_overview(
            config, pipeline, display_format, mode="evaluation"
        )

        # Handle CLI flag overrides for SHAP and visualizations
        if hasattr(args, "interpret") and args.interpret:
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
        with suppress_stderr():
            with suppress_stderr():
                from timeflies.core import PipelineManager

        # Initialize pipeline and run evaluation-only workflow
        # (This includes metrics, interpretation, and visualizations based on config)
        pipeline = PipelineManager(config, mode="evaluation")
        pipeline.run_evaluation()

        # Restore original config settings if overridden
        if original_shap is not None:
            config.interpretation.shap.enabled = original_shap
        if original_viz is not None:
            config.visualizations.enabled = original_viz

        # Run analysis after evaluation if requested
        if hasattr(args, "with_analysis") and args.with_analysis:
            from .analysis import analyze_command

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
