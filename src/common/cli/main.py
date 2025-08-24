#!/usr/bin/env python3
"""Main CLI entry point for TimeFlies package."""

import os
import sys
from pathlib import Path


def main_cli(argv=None):
    """Main entry point for CLI - returns exit code.

    Args:
        argv: Optional list of arguments. If None, uses sys.argv
    """
    # Import and run the CLI
    from common.cli.commands import execute_command
    from common.cli.parser import create_main_parser

    parser = create_main_parser()
    args = parser.parse_args(argv)

    # Execute the command
    success = execute_command(args)
    return 0 if success else 1


def main():
    """Main entry point for timeflies command."""
    # For installed packages, no path manipulation needed
    # For development, add src to Python path
    try:
        # Try to import - if it works, we're in an installed package
        from common.cli.parser import create_main_parser
    except ImportError:
        # We're in development mode, add src to path
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))

    exit_code = main_cli()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
