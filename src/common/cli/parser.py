"""
TimeFlies CLI Parser

Modern subcommand-based CLI parser for TimeFlies.
"""

import argparse
from typing import Optional


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        description="TimeFlies v1.0: Machine Learning for Aging Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
User Workflow:
  # Complete workflow (recommended)
  python run_timeflies.py setup [--batch-correct]    # Split data + verify + create dirs
  python run_timeflies.py train [--with-eda]         # Train models with evaluation
  python run_timeflies.py evaluate [--with-eda]      # Evaluate models on test data
  python run_timeflies.py analyze [--with-eda]       # Project-specific analysis scripts

  # Individual steps
  python run_timeflies.py split                      # Just create train/eval splits
  python run_timeflies.py eda --save-report          # Exploratory data analysis
  python run_timeflies.py batch-correct              # Apply batch correction
  python run_timeflies.py verify                     # Check system status

  # Development/testing
  python run_timeflies.py test [unit|integration]    # Run test suite
  python run_timeflies.py test --coverage            # Generate coverage report
  python run_timeflies.py create-test-data           # Generate test fixtures
  python run_timeflies.py update                     # Update to latest version

  # Project switching (temporary override)
  python run_timeflies.py --aging train              # Train aging project
  python run_timeflies.py --alzheimers analyze       # Analyze Alzheimer's project
  python run_timeflies.py --tissue head train        # Override tissue type

  # Global options work with any command
  --batch-corrected --verbose --tissue head --aging

  # Permanent project switching: Edit configs/default.yaml
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

    # Split command (create train/eval data splits)
    subparsers.add_parser(
        "split", help="Create train/eval data splits from your original data"
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

    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_main_parser()
    return parser.parse_args(argv)
