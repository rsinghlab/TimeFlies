"""
TimeFlies CLI Analysis Commands

Contains EDA and analysis commands for data exploration and model analysis.
"""

from ._utils import suppress_stderr


def eda_command(args, config) -> int:
    """Run exploratory data analysis on the dataset."""
    import os
    from pathlib import Path

    from ...analysis.eda import EDAHandler

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO and WARNING

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


def analyze_command(args, config) -> int:
    """Run project-specific analysis on a trained model."""
    # Suppress TensorFlow warnings for cleaner output
    import os
    from pathlib import Path

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO and WARNING

    try:
        print("=" * 60)
        print("🔬 TIMEFLIES ANALYSIS")
        print("=" * 60)
        print(
            f"Project: {getattr(config, 'project', 'unknown').replace('_', ' ').title()}"
        )
        print(f"Tissue: {config.data.tissue.title()}")
        print(f"Model: {config.data.model}")
        print(f"Target: {config.data.target_variable.title()}")
        print("-" * 60)

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
            from timeflies.utils.path_manager import PathManager

            path_manager = PathManager(config)

            # Try to find best model for current config
            try:
                best_model_dir = path_manager.get_best_model_dir_for_config()
                predictions_path = (
                    Path(best_model_dir) / "evaluation" / "predictions.csv"
                )
                if predictions_path.exists():
                    print("✅ Found predictions from best model")
                else:
                    predictions_path = None
            except Exception:
                predictions_path = None

        if predictions_path and predictions_path.exists():
            print("📊 Running analysis on predictions from best model...")

            # Just run the analysis script without reloading everything
            with suppress_stderr():
                from timeflies.core import PipelineManager

            pipeline = PipelineManager(config)

            # Pass the found predictions path to the pipeline
            pipeline._analysis_predictions_path = str(predictions_path)

            # Only run the analysis script part
            if hasattr(pipeline, "run_analysis_script"):
                pipeline.run_analysis_script()

            print("\n[OK] Analysis completed successfully!")
            return 0

        print("WARNING:  No predictions found, need to generate them...")

        # Check if model exists
        from timeflies.utils.path_manager import PathManager

        path_manager = PathManager(config)
        model_dir = path_manager.construct_model_directory()
        model_path = Path(model_dir) / "model.h5"

        if not model_path.exists():
            print(f"WARNING:  No trained model found at {model_path}")
            print("PACKAGE: Training model first...")

            # Run training
            with suppress_stderr():
                from timeflies.core import PipelineManager

            pipeline = PipelineManager(config)
            pipeline.run_training()
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
            with suppress_stderr():
                from timeflies.core import PipelineManager

            # Initialize pipeline and run evaluation with analysis
            pipeline = PipelineManager(config, mode="evaluation")
            pipeline.run_evaluation()

            print("\n[OK] Analysis completed successfully!")
            return 0

        finally:
            # Restore original setting
            config.analysis.run_analysis_script.enabled = original_enabled

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        # Always print traceback for debugging
        import traceback

        traceback.print_exc()
        return 1
