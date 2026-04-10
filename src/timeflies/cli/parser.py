"""
TimeFlies CLI Parser

Modern subcommand-based CLI parser for TimeFlies.
"""

import argparse


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        description="TimeFlies v1.0: Machine Learning for Single-Cell Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
User Workflow:
  # Complete workflow (recommended)
  timeflies setup [--batch-correct]    # Split data + verify + create dirs
  timeflies train [--with-eda]         # Train models with evaluation
  timeflies evaluate [--with-eda]      # Evaluate models on test data
  timeflies analyze [--with-eda]       # Project-specific analysis scripts

  # Individual steps
  timeflies split                      # Just create train/eval splits
  timeflies eda --save-report          # Exploratory data analysis
  timeflies batch-correct              # Apply batch correction
  timeflies verify                     # Check system status

  # Development/testing
  timeflies test [unit|integration]    # Run test suite
  timeflies test --coverage            # Generate coverage report
  timeflies create-test-data           # Generate test fixtures

  # Automated model training systems
  timeflies tune                       # Run hyperparameter tuning
  timeflies queue                      # Run default model queue

  # Project switching (temporary override)
  timeflies --project fruitfly_aging train
  timeflies --tissue head train

  # Global options work with any command
  --batch-corrected --verbose --tissue head --project my_project
        """,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--batch-corrected",
        action="store_true",
        help="Use batch-corrected data for all operations",
    )

    # Common overrides
    parser.add_argument(
        "--tissue", type=str, help="Override tissue type (e.g., head, body)"
    )
    parser.add_argument(
        "--model", type=str, help="Override model type (e.g., CNN, MLP, xgboost)"
    )
    parser.add_argument(
        "--target", type=str, help="Override target variable (e.g., age)"
    )

    # Project selection
    project_group = parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--project",
        type=str,
        dest="project",
        help="Project name (e.g., fruitfly_aging, fruitfly_alzheimers, or custom)",
    )
    project_group.add_argument(
        "--aging",
        action="store_const",
        const="fruitfly_aging",
        dest="project",
        help="Shorthand for --project fruitfly_aging",
    )
    project_group.add_argument(
        "--alzheimers",
        "--alz",
        action="store_const",
        const="fruitfly_alzheimers",
        dest="project",
        help="Shorthand for --project fruitfly_alzheimers",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model using project config settings (includes auto-evaluation)",
    )
    train_parser.add_argument(
        "--with-eda", action="store_true", help="Run EDA before training"
    )
    train_parser.add_argument(
        "--with-analysis", action="store_true", help="Run analysis after training"
    )

    # EDA command
    eda_parser = subparsers.add_parser(
        "eda", help="Run exploratory data analysis on the full dataset"
    )
    eda_parser.add_argument(
        "--save-report",
        action="store_true",
        help="Generate HTML report of EDA results",
    )

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Complete setup: split data + verify system + create directories"
    )
    setup_parser.add_argument(
        "--batch-correct",
        action="store_true",
        help="Include batch correction in setup workflow",
    )
    setup_parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force recreate data splits even if they already exist",
    )

    # Split command
    split_parser = subparsers.add_parser(
        "split", help="Create train/eval data splits from your original data"
    )
    split_parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force recreate data splits even if they already exist",
    )

    # Verify command
    subparsers.add_parser("verify", help="Verify installation and system setup")

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Run test suite with various options"
    )
    test_parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "functional", "system", "all"],
        help="Type of tests to run (default: all)",
    )
    test_parser.add_argument(
        "--coverage", action="store_true", help="Generate HTML coverage report"
    )
    test_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed test output"
    )
    test_parser.add_argument(
        "--fast",
        action="store_true",
        help="Run unit + integration only (skip slow tests)",
    )
    test_parser.add_argument(
        "--debug",
        action="store_true",
        help="Stop on first failure with detailed output",
    )
    test_parser.add_argument(
        "--rerun", action="store_true", help="Re-run failed tests only"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model using project config settings"
    )
    eval_parser.add_argument(
        "--with-eda", action="store_true", help="Run EDA before evaluation"
    )
    eval_parser.add_argument(
        "--with-analysis", action="store_true", help="Run analysis after evaluation"
    )
    eval_parser.add_argument(
        "--interpret",
        action="store_true",
        help="Enable SHAP interpretation (overrides config setting)",
    )
    eval_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualizations (overrides config setting)",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run project-specific analysis on trained model"
    )
    analyze_parser.add_argument(
        "--predictions-path",
        type=str,
        help="Path to existing predictions CSV (skip model loading)",
    )
    analyze_parser.add_argument(
        "--with-eda", action="store_true", help="Run EDA before analysis"
    )
    analyze_parser.add_argument(
        "--analysis-script",
        type=str,
        help="Path to custom analysis script (Python file with run_analysis function)",
    )

    # Batch correction command
    subparsers.add_parser(
        "batch-correct", help="Run batch correction using project config settings"
    )

    # Create test data command
    test_data_parser = subparsers.add_parser(
        "create-test-data",
        help="Create test data fixtures using 3-tier strategy",
    )
    test_data_parser.add_argument(
        "--tier",
        type=str,
        choices=["tiny", "synthetic", "real", "all"],
        default="all",
        help="Which tier of test data to create (default: all)",
    )
    test_data_parser.add_argument(
        "--cells",
        type=int,
        help="Override number of cells (tiny: 50, synthetic: 500, real: 5000)",
    )
    test_data_parser.add_argument(
        "--genes",
        type=int,
        help="Override number of genes (tiny: 100, synthetic: 1000, real: 2000)",
    )
    test_data_parser.add_argument(
        "--batch-versions",
        action="store_true",
        help="Create both batch-corrected and uncorrected versions",
    )

    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser(
        "tune",
        help="Run automated hyperparameter tuning",
    )
    tune_parser.add_argument(
        "config",
        nargs="?",
        default="configs/default.yaml",
        help="Path to configuration YAML file (default: configs/default.yaml)",
    )
    tune_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if checkpoint exists",
    )

    # Queue command
    queue_parser = subparsers.add_parser(
        "queue",
        help="Run automated sequential model training from queue configuration",
    )
    queue_parser.add_argument(
        "config",
        nargs="?",
        default="configs/model_queue.yaml",
        help="Path to queue configuration YAML file (default: configs/model_queue.yaml)",
    )
    queue_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if checkpoint exists",
    )
    queue_parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run analysis queue only (skip training)",
    )

    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_main_parser()
    return parser.parse_args(argv)
