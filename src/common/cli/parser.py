"""
TimeFlies CLI Parser

Modern subcommand-based CLI parser for TimeFlies.
"""

import argparse


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        description="TimeFlies v1.0: Machine Learning for Aging Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
User Workflow:
  # Complete workflow (recommended)
  timeflies setup [--batch-correct]    # Split data + verify + create dirs
  timeflies train [--with-eda]         # Train models with evaluation
  timeflies evaluate [--with-eda]      # Evaluate models on test data
  timeflies analyze [--with-eda]       # Project-specific analysis scripts

  # Graphical interface (web-based)
  timeflies gui                        # Launch web GUI in browser

  # Individual steps
  timeflies split                      # Just create train/eval splits
  timeflies eda --save-report          # Exploratory data analysis
  timeflies batch-correct              # Apply batch correction
  timeflies verify                     # Check system status

  # Development/testing
  timeflies test [unit|integration]    # Run test suite
  timeflies test --coverage            # Generate coverage report
  timeflies create-test-data           # Generate test fixtures
  timeflies update                     # Update to latest version

  # Automated model training systems
  timeflies tune                       # Run hyperparameter tuning (uses config.yaml)
  timeflies tune custom_config.yaml    # Run with custom config file
  timeflies queue                      # Run default model queue
  timeflies queue custom_queue.yaml    # Run custom queue configuration

  # Project switching (temporary override)
  timeflies --aging train              # Train aging project
  timeflies --alzheimers analyze       # Analyze Alzheimer's project
  timeflies --tissue head train        # Override tissue type

  # Global options work with any command
  --batch-corrected --verbose --tissue head --aging

  # Permanent project switching: Edit config.yaml
        """,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Batch correction flag (global)
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

    # Project selection (mutually exclusive)
    project_group = parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--aging",
        action="store_const",
        const="fruitfly_aging",
        dest="project",
        help="Use fruitfly_aging project (healthy flies)",
    )
    project_group.add_argument(
        "--alzheimers",
        "--alz",
        action="store_const",
        const="fruitfly_alzheimers",
        dest="project",
        help="Use fruitfly_alzheimers project (disease models)",
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

    # EDA command (simplified)
    eda_parser = subparsers.add_parser(
        "eda", help="Run exploratory data analysis on the full dataset"
    )
    eda_parser.add_argument(
        "--save-report", action="store_true", help="Generate HTML report of EDA results"
    )

    # Setup command (main user workflow)
    setup_parser = subparsers.add_parser(
        "setup", help="Complete setup: split data + verify system + create directories"
    )
    setup_parser.add_argument(
        "--batch-correct",
        action="store_true",
        help="Include batch correction in setup workflow",
    )
    setup_parser.add_argument(
        "--dev",
        action="store_true",
        help="Developer setup: only create environments (.venv and .venv_batch)",
    )
    setup_parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing development environment with latest dependencies",
    )
    setup_parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force recreate data splits even if they already exist",
    )

    # Split command (create train/eval data splits)
    split_parser = subparsers.add_parser(
        "split", help="Create train/eval data splits from your original data"
    )
    split_parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force recreate data splits even if they already exist",
    )

    # Verify command (system setup verification)
    subparsers.add_parser("verify", help="Verify installation and system setup")

    # Test command (test runner integration)
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

    # Batch correction command (no flags - uses project config)
    subparsers.add_parser(
        "batch-correct", help="Run batch correction using project config settings"
    )

    # Create test data command (3-tier strategy)
    test_data_parser = subparsers.add_parser(
        "create-test-data",
        help="Create test data fixtures using 3-tier strategy: tiny real + synthetic + dev real data",
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

    # Update command
    subparsers.add_parser(
        "update",
        help="Update TimeFlies to the latest version from GitHub",
    )

    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser(
        "tune",
        help="Run automated hyperparameter tuning with grid, random, or Bayesian optimization",
        description="Optimize hyperparameters for TimeFlies models using grid search, "
        "random search, or Bayesian optimization (Optuna). Supports model architecture "
        "exploration and comprehensive hyperparameter optimization with progress tracking.",
    )
    tune_parser.add_argument(
        "config",
        nargs="?",
        default="configs/default.yaml",
        help="Path to configuration YAML file with hyperparameter tuning enabled (default: configs/default.yaml)",
    )
    tune_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if checkpoint exists (default: resume from checkpoint)",
    )

    # Queue command for automated multi-model training
    queue_parser = subparsers.add_parser(
        "queue",
        help="Run automated sequential model training from queue configuration",
        description="Train multiple models sequentially with different configurations. "
        "Supports checkpoints, progress tracking, and comprehensive comparison reports. "
        "Queue configurations define multiple models with hyperparameters and settings.",
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
        help="Start fresh even if checkpoint exists (default: resume from checkpoint)",
    )

    # GUI command for web interface
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch web-based graphical user interface",
        description="Start a web-based GUI for TimeFlies in your browser. "
        "Provides point-and-click access to all TimeFlies functionality including "
        "setup, training, batch correction, and hyperparameter tuning. "
        "No system dependencies required - works in any browser.",
    )
    gui_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number for web server (default: 7860)",
    )
    gui_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for web server (default: 127.0.0.1 - local only)",
    )
    gui_parser.add_argument(
        "--share",
        action="store_true",
        help="Create public URL for remote access (use with caution)",
    )
    gui_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for development",
    )

    # Uninstall command
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Uninstall TimeFlies and clean up installation",
        description="""
Remove TimeFlies installation including virtual environments,
source code, and optionally data directories. Use with caution
as this action cannot be undone.
        """.strip(),
    )
    uninstall_parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep data, outputs, models, and config directories",
    )
    uninstall_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    uninstall_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing it",
    )

    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_main_parser()
    return parser.parse_args(argv)
