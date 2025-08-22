#!/usr/bin/env python3
"""Main CLI entry point for TimeFlies package."""

import sys
import os
from pathlib import Path

def main():
    """Main entry point for timeflies command."""
    # Add src to Python path for imports
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    
    # Import and run the CLI
    from shared.cli.parser import create_parser
    from shared.cli.commands import execute_command
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Execute the command
    success = execute_command(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()