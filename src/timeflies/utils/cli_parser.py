"""Command line interface parser for TimeFlies project."""

import argparse
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..utils.constants import SUPPORTED_MODEL_TYPES, SUPPORTED_ENCODING_VARS, VALID_SEX_TYPES
from ..utils.logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for TimeFlies."""
    
    parser = argparse.ArgumentParser(
        description="TimeFlies: Time-series analysis for biological data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python scripts/run_pipeline.py
  
  # Run with custom config file
  python scripts/run_pipeline.py --config configs/my_config.yaml
  
  # Override specific parameters
  python scripts/run_pipeline.py --model-type CNN --encoding-variable age
  
  # Train only (no evaluation)
  python scripts/run_pipeline.py --train-only
  
  # Evaluate existing model
  python scripts/run_pipeline.py --evaluate-only --model-path Models/my_model/
        """
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file (default: configs/default.yaml)'
    )
    config_group.add_argument(
        '--save-config',
        type=str,
        help='Save effective configuration to specified path'
    )
    
    # Data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        '--tissue',
        choices=['head', 'body', 'all'],
        help='Tissue type to analyze'
    )
    data_group.add_argument(
        '--model-type',
        choices=SUPPORTED_MODEL_TYPES,
        help='Model type to use'
    )
    data_group.add_argument(
        '--encoding-variable',
        choices=SUPPORTED_ENCODING_VARS,
        help='Variable to encode/predict'
    )
    data_group.add_argument(
        '--cell-type',
        type=str,
        help='Cell type to analyze (default: all)'
    )
    data_group.add_argument(
        '--sex-type',
        choices=VALID_SEX_TYPES,
        help='Sex type to analyze'
    )
    data_group.add_argument(
        '--batch-correction',
        action='store_true',
        help='Enable batch correction'
    )
    data_group.add_argument(
        '--no-batch-correction',
        action='store_true',
        help='Disable batch correction'
    )
    
    # Gene preprocessing
    gene_group = parser.add_argument_group('Gene Preprocessing')
    gene_group.add_argument(
        '--hvg',
        action='store_true',
        help='Use highly variable genes'
    )
    gene_group.add_argument(
        '--remove-sex-genes',
        action='store_true',
        help='Remove sex-linked genes'
    )
    gene_group.add_argument(
        '--remove-autosomal-genes',
        action='store_true',
        help='Remove autosomal genes'
    )
    gene_group.add_argument(
        '--only-lnc-genes',
        action='store_true',
        help='Keep only lncRNA genes'
    )
    gene_group.add_argument(
        '--balance-genes',
        action='store_true',
        help='Balance autosomal and sex genes'
    )
    
    # Model training
    training_group = parser.add_argument_group('Model Training')
    training_group.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    training_group.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    training_group.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate for training'
    )
    training_group.add_argument(
        '--validation-split',
        type=float,
        help='Validation split ratio (0.0-1.0)'
    )
    
    # Pipeline control
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument(
        '--train-only',
        action='store_true',
        help='Only train the model (skip evaluation)'
    )
    pipeline_group.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate existing model (skip training)'
    )
    pipeline_group.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Only run preprocessing (skip training and evaluation)'
    )
    pipeline_group.add_argument(
        '--model-path',
        type=str,
        help='Path to existing model for evaluation'
    )
    
    # Logging and output
    output_group = parser.add_argument_group('Logging and Output')
    output_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    output_group.add_argument(
        '--log-file',
        type=str,
        help='Log to file (in addition to console)'
    )
    output_group.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (equivalent to --log-level DEBUG)'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet output (equivalent to --log-level WARNING)'
    )
    
    # Utilities
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit (no execution)'
    )
    util_group.add_argument(
        '--gpu',
        action='store_true',
        help='Force GPU usage'
    )
    util_group.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    util_group.add_argument(
        '--random-state',
        type=int,
        help='Random state for reproducibility'
    )
    
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args(args)


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments for consistency."""
    
    # Check conflicting options
    if args.train_only and args.evaluate_only:
        raise ValueError("Cannot specify both --train-only and --evaluate-only")
    
    if args.batch_correction and args.no_batch_correction:
        raise ValueError("Cannot specify both --batch-correction and --no-batch-correction")
    
    if args.verbose and args.quiet:
        raise ValueError("Cannot specify both --verbose and --quiet")
    
    if args.gpu and args.cpu:
        raise ValueError("Cannot specify both --gpu and --cpu")
    
    # Check file paths
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    if args.evaluate_only and args.model_path and not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    # Validate ranges
    if args.validation_split and not (0.0 <= args.validation_split <= 1.0):
        raise ValueError("Validation split must be between 0.0 and 1.0")
    
    if args.epochs and args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if args.batch_size and args.batch_size <= 0:
        raise ValueError("Batch size must be positive")


def args_to_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed arguments to configuration overrides."""
    
    overrides = {}
    
    # Data parameters
    if args.tissue:
        overrides['data.tissue'] = args.tissue
    if args.model_type:
        overrides['data.model_type'] = args.model_type
    if args.encoding_variable:
        overrides['data.encoding_variable'] = args.encoding_variable
    if args.cell_type:
        overrides['data.cell_type'] = args.cell_type
    if args.sex_type:
        overrides['data.sex_type'] = args.sex_type
    
    # Batch correction
    if args.batch_correction:
        overrides['data.batch_correction.enabled'] = True
    elif args.no_batch_correction:
        overrides['data.batch_correction.enabled'] = False
    
    # Gene preprocessing
    if args.hvg:
        overrides['gene_preprocessing.gene_filtering.highly_variable_genes'] = True
    if args.remove_sex_genes:
        overrides['gene_preprocessing.gene_filtering.remove_sex_genes'] = True
    if args.remove_autosomal_genes:
        overrides['gene_preprocessing.gene_filtering.remove_autosomal_genes'] = True
    if args.only_lnc_genes:
        overrides['gene_preprocessing.gene_filtering.only_keep_lnc_genes'] = True
    if args.balance_genes:
        overrides['gene_preprocessing.gene_balancing.balance_genes'] = True
    
    # Model training
    if args.epochs:
        overrides['model.training.epochs'] = args.epochs
    if args.batch_size:
        overrides['model.training.batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['model.training.learning_rate'] = args.learning_rate
    if args.validation_split:
        overrides['model.training.validation_split'] = args.validation_split
    
    # General
    if args.random_state:
        overrides['general.random_state'] = args.random_state
    
    # Device
    if args.gpu:
        overrides['device.processor'] = 'GPU'
    elif args.cpu:
        overrides['device.processor'] = 'CPU'
    
    return overrides


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Set up logging based on command line arguments."""
    
    # Determine log level
    if args.verbose:
        log_level = 'DEBUG'
    elif args.quiet:
        log_level = 'WARNING'
    else:
        log_level = args.log_level
    
    # Set up logging
    setup_logging(
        level=log_level,
        log_file=args.log_file,
        log_dir="logs" if args.log_file else None,
        format_style="detailed"
    )