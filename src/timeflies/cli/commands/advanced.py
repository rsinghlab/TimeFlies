"""
TimeFlies CLI Advanced Commands

Contains hyperparameter tuning and model queue management commands.
"""


def tune_command(args) -> int:
    """
    Run hyperparameter tuning with grid, random, or Bayesian optimization.

    Args:
        args: Command line arguments containing tuning config path

    Returns:
        0 on success, 1 on failure
    """
    from pathlib import Path

    from timeflies.core.hyperparameter_tuner import HyperparameterTuner

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

    from timeflies.core.model_queue import ModelQueueManager

    print("\n" + "=" * 60)
    print("TIMEFLIES MODEL QUEUE MANAGER")
    print("=" * 60)

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"[ERROR] Queue configuration not found: {config_path}")
        print("\nExample queue configuration: configs/model_queue_example.yaml")
        return 1

    try:
        # Check if this is analysis-only mode
        if hasattr(args, "analysis") and args.analysis:
            from timeflies.core.analysis_queue import AnalysisQueueRunner

            print("Running analysis queue only (skip training)")
            print(f"Queue config: {config_path}")
            print("")

            # Create analysis queue runner
            analysis_runner = AnalysisQueueRunner()

            # Load queue config to get model list for filtering
            import yaml

            with open(config_path) as f:
                queue_config = yaml.safe_load(f)

            # Check if this is an analysis queue config
            if "models_to_analyze" in queue_config:
                # Analysis queue format - explicit model list
                models_list = queue_config["models_to_analyze"]
                print(
                    f"Using analysis queue config with {len(models_list)} explicit models"
                )
                print(
                    f"Models to analyze: {models_list[:3]}{'...' if len(models_list) > 3 else ''}"
                )

                # Use the AnalysisQueueRunner with the specific models
                analysis_runner.run_queue_with_models(
                    model_list=models_list,
                    analysis_script=queue_config.get("analysis_settings", {}).get(
                        "analysis_script", None
                    ),
                )
                return 0

            else:
                # Model training queue config - analyze all available models
                print(
                    "Using model training queue config, analyzing all available models"
                )
                pattern = "*"

            # Run analysis queue
            analysis_runner.run_queue(
                model_pattern=pattern,
                analysis_script=None,  # Use default analysis script
            )

            return 0
        else:
            # Normal training queue mode
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
