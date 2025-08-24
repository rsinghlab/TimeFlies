#!/usr/bin/env python3
"""Simple CLI wrapper for TimeFlies that works with setuptools."""


def main():
    """Entry point that works with installed packages."""
    try:
        # Import the actual CLI components
        from common.cli.commands import execute_command
        from common.cli.parser import create_main_parser

        # Run the CLI
        parser = create_main_parser()
        args = parser.parse_args()

        # Execute the command
        success = execute_command(args)
        exit_code = 0 if success else 1
        exit(exit_code)

    except ImportError as e:
        print(f"Error importing TimeFlies modules: {e}")
        print("Make sure TimeFlies is properly installed.")
        exit(1)


if __name__ == "__main__":
    main()
