#!/usr/bin/env python3
"""Main CLI entry point for TimeFlies package."""

import sys


def main_cli(argv=None):
    """Main entry point for CLI - returns exit code.

    Args:
        argv: Optional list of arguments. If None, uses sys.argv
    """
    from timeflies.cli.commands import execute_command
    from timeflies.cli.parser import create_main_parser

    parser = create_main_parser()
    args = parser.parse_args(argv)

    success = execute_command(args)
    return 0 if success else 1


def main():
    """Main entry point for timeflies command."""
    exit_code = main_cli()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
