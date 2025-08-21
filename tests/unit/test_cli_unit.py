"""Unit tests for CLI modules."""

import pytest
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.cli.parser import create_main_parser


class TestCLIParser:
    """Test CLI argument parsing."""
    
    def test_create_parser_basic(self):
        """Test parser creation."""
        parser = create_main_parser()
        assert parser is not None
        assert hasattr(parser, 'parse_args')
    
    def test_parser_help(self):
        """Test parser help doesn't crash."""
        parser = create_main_parser()
        with pytest.raises(SystemExit):  # Help exits with code 0
            parser.parse_args(['--help'])
    
    def test_parser_train_command(self):
        """Test train command parsing."""
        parser = create_main_parser()
        args = parser.parse_args(['train'])
        assert hasattr(args, 'command')
    
    def test_parser_evaluate_command(self):
        """Test evaluate command parsing."""
        parser = create_main_parser()
        args = parser.parse_args(['evaluate'])
        assert hasattr(args, 'command')
    
    def test_parser_test_command(self):
        """Test test command parsing."""
        parser = create_main_parser()
        args = parser.parse_args(['test'])
        assert hasattr(args, 'command')


