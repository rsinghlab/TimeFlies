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
Examples:
  # Setup data (run this first)
  python run_setup.py
  
  # Train a CNN model for age prediction
  python run_timeflies.py train --tissue head --model cnn --target age
  
  # Train with batch correction
  python run_timeflies.py train --tissue head --model cnn --batch-correction
  
  # Test installation
  python run_timeflies.py test
        """
    )
    
    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--tissue', choices=['head', 'body', 'all'], default='head')
    train_parser.add_argument('--model', choices=['cnn', 'mlp', 'xgboost', 'random_forest', 'logistic'], default='cnn')
    train_parser.add_argument('--target', choices=['age', 'sex', 'tissue'], default='age')
    train_parser.add_argument('--batch-correction', action='store_true', help='Use batch correction')
    train_parser.add_argument('--cell-type', default='all', help='Cell type filter')
    train_parser.add_argument('--sex-type', choices=['all', 'male', 'female'], default='all')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test installation')
    
    # Evaluate command  
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model-path', required=True, help='Path to trained model')
    eval_parser.add_argument('--tissue', choices=['head', 'body', 'all'], default='head')
    
    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_main_parser()
    return parser.parse_args(argv)