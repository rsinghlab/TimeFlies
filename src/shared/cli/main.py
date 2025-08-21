"""
TimeFlies CLI Main Entry Point

This module provides the main CLI interface that integrates with the existing
cli_parser.py to provide a clean, modular command structure.
"""

import sys
from typing import Optional

from ..core.active_config import get_config_for_active_project, get_active_project
from ..utils.logging_config import setup_logging
from .parser import parse_arguments
from .commands import (
    train_command, 
    evaluate_command,
    run_system_tests,
    run_test_suite,
    setup_command,
    batch_command,
    create_test_data_command
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
        
        # Execute commands that don't need config immediately
        if args.command == 'setup':
            return setup_command(args)
        elif args.command == 'create-test-data':
            return create_test_data_command(args)
        elif args.command == 'verify':
            return run_system_tests(args)
        
        # Load configuration (auto-detect active project)
        try:
            # Auto-detect active project
            active_project = get_active_project()
            print(f"Active project: {active_project}")
            
            # Load config for active project 
            config_manager = get_config_for_active_project("default")
            config = config_manager.get_config()
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Please check your configs/default.yaml file")
            return 1
        
        # Execute based on subcommand
        if args.command == 'train':
            return train_command(args, config)
        elif args.command == 'evaluate':
            return evaluate_command(args, config)
        elif args.command == 'verify':
            return run_system_tests(args)
        elif args.command == 'test':
            return run_test_suite(args)
        elif args.command == 'batch-correct':
            return batch_command(args)
        elif args.command == 'create-test-data':
            return create_test_data_command(args)
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




