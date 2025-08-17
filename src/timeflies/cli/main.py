"""
TimeFlies CLI Main Entry Point

This module provides the main CLI interface that integrates with the existing
cli_parser.py to provide a clean, modular command structure.
"""

import sys
from typing import Optional

from ..core.config_manager import ConfigManager
from ..utils.logging_config import setup_logging
from .parser import parse_arguments
from .commands import (
    train_command, 
    evaluate_command,
    test_command
)


def main_cli(argv: Optional[list] = None) -> int:
    """
    Main CLI entry point for TimeFlies.
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse arguments using new subcommand parser
        args = parse_arguments(argv)
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(level=log_level)
        
        # Load configuration
        if args.config:
            config_manager = ConfigManager(args.config)
        else:
            config_manager = ConfigManager()  # Uses default config
        
        config = config_manager.get_config()
        
        # Execute based on subcommand
        if args.command == 'train':
            return train_command(args, config)
        elif args.command == 'evaluate':
            return evaluate_command(args, config)
        elif args.command == 'test':
            return test_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1




