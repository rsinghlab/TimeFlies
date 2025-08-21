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
  
  # Development/testing
  python run_timeflies.py test               # Run all tests
  python run_timeflies.py test unit --fast   # Quick unit tests
  python run_timeflies.py test --coverage    # Run with coverage
  python run_timeflies.py test --rerun       # Re-run failed tests
  
  # Project switching: Edit configs/default.yaml and change 'project: fruitfly_aging'
        """
    )
    
    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model using project config settings')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up data and directories')
    
    # Verify command (system setup verification)
    verify_parser = subparsers.add_parser('verify', help='Verify installation and system setup')
    
    # Test command (test runner integration)
    test_parser = subparsers.add_parser('test', help='Run test suite with various options')
    test_parser.add_argument('test_type', nargs='?', default='all', 
                           choices=['unit', 'integration', 'functional', 'system', 'all'],
                           help='Type of tests to run (default: all)')
    test_parser.add_argument('--coverage', action='store_true', help='Generate HTML coverage report')
    test_parser.add_argument('--fast', action='store_true', help='Run unit + integration only (skip slow tests)')
    test_parser.add_argument('--debug', action='store_true', help='Stop on first failure with detailed output')
    test_parser.add_argument('--rerun', action='store_true', help='Re-run failed tests only')
    
    
    # Evaluate command  
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model using project config settings')
    
    # Batch correction command (no flags - uses project config)
    batch_parser = subparsers.add_parser('batch-correct', help='Run batch correction using project config settings')
    
    # Create test data command
    test_data_parser = subparsers.add_parser('create-test-data', help='Create test data fixtures by sampling from all available project data')
    
    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_main_parser()
    return parser.parse_args(argv)