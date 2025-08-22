#!/usr/bin/env python3
"""Main CLI entry point for TimeFlies package."""

import sys
import os
from pathlib import Path

def main_cli():
    """Main entry point for CLI - returns exit code."""
    # Import and run the CLI
    from common.cli.parser import create_main_parser
    from common.cli.commands import execute_command
    
    parser = create_main_parser()
    args = parser.parse_args()
    
    # Execute the command
    success = execute_command(args)
    return 0 if success else 1

def main():
    """Main entry point for timeflies command."""
    # Add src to Python path for imports
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    
    exit_code = main_cli()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()