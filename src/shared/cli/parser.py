"""
TimeFlies CLI Parser

Modern subcommand-based CLI parser for TimeFlies.
"""

import argparse
from typing import List, Optional


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        description="TimeFlies: Machine Learning for Aging Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  # Initial setup (run once)
  bash setup_dev_env.sh                      # Set up environment
  python run_timeflies.py verify             # Verify installation
  python run_timeflies.py setup              # Split data into train/eval
  
  # Main workflow (auto-detects active project from configs/default.yaml)  
  python run_timeflies.py train              # Train models
  python run_timeflies.py evaluate           # Evaluate with SHAP/visualization
  python run_timeflies.py analyze            # Run project-specific analysis
  
  # Project switching (temporary override)
  python run_timeflies.py --aging train      # Train aging project
  python run_timeflies.py --alzheimers train # Train Alzheimer's project
  python run_timeflies.py --alz verify       # Quick verify on Alzheimer's
  
  # Development/testing
  python run_timeflies.py test               # Run all tests
  python run_timeflies.py test unit          # Run unit tests only
  python run_timeflies.py test integration   # Run integration tests only
  python run_timeflies.py test --fast        # Quick tests (unit + integration)
  python run_timeflies.py test --coverage    # Generate HTML coverage report
  python run_timeflies.py test --debug       # Stop on first failure with details
  python run_timeflies.py test --rerun       # Re-run failed tests only
  
  # Permanent project switching: Edit configs/default.yaml and change 'project: fruitfly_aging'
        """,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    
    # Batch correction flag (global)
    parser.add_argument(
        "--batch-corrected", action="store_true", 
        help="Use batch-corrected data for all operations"
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
        "--aging", action="store_const", const="fruitfly_aging", dest="project",
        help="Use fruitfly_aging project (healthy flies)"
    )
    project_group.add_argument(
        "--alzheimers", "--alz", action="store_const", const="fruitfly_alzheimers", dest="project",
        help="Use fruitfly_alzheimers project (disease models)"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train a model using project config settings (includes auto-evaluation)"
    )
    train_parser.add_argument(
        "--with-eda", action="store_true", help="Run EDA before training"
    )
    train_parser.add_argument(
        "--with-analysis", action="store_true", help="Run analysis after training"
    )
    
    # EDA command (new)
    eda_parser = subparsers.add_parser(
        "eda", help="Run exploratory data analysis on the dataset"
    )
    eda_parser.add_argument(
        "--split", type=str, choices=["all", "train", "test"], default="all",
        help="Which data split to analyze (default: all)"
    )
    eda_parser.add_argument(
        "--save-report", action="store_true", 
        help="Generate HTML report of EDA results"
    )

    # Unified setup command (new!)
    unified_setup_parser = subparsers.add_parser(
        "setup-all", help="🚀 One-command setup: verify + test-data + splits + directories (RECOMMENDED)"
    )
    unified_setup_parser.add_argument(
        "--skip-batch", action="store_true",
        help="Skip batch correction setup"
    )
    
    # Setup command (individual steps)
    setup_parser = subparsers.add_parser("setup", help="Set up data and directories only")

    # Verify command (system setup verification)
    verify_parser = subparsers.add_parser(
        "verify", help="Verify installation and system setup"
    )

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

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run project-specific analysis on trained model"
    )
    analyze_parser.add_argument(
        "--predictions-path", type=str,
        help="Path to existing predictions CSV (skip model loading)"
    )
    analyze_parser.add_argument(
        "--with-eda", action="store_true", 
        help="Run EDA before analysis"
    )

    # Batch correction command (no flags - uses project config)
    batch_parser = subparsers.add_parser(
        "batch-correct", help="Run batch correction using project config settings"
    )

    # Create test data command
    test_data_parser = subparsers.add_parser(
        "create-test-data",
        help="Create test data fixtures by sampling from all available project data",
    )

    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_main_parser()
    return parser.parse_args(argv)
