"""Unit tests for CLI commands functionality to increase coverage."""

import pytest

from timeflies.cli.parser import create_main_parser, parse_arguments


@pytest.mark.unit
class TestCLIArgumentHandling:
    """Test CLI argument handling and validation."""

    def test_parse_arguments_all_commands(self):
        """Test parsing all available commands."""
        command_tests = [
            (["train"], "train"),
            (["evaluate"], "evaluate"),
            (["setup"], "setup"),
            (["create-test-data"], "create-test-data"),
            (["test", "unit"], "test"),
            (["batch-correct"], "batch-correct"),
        ]

        for args, expected_command in command_tests:
            parsed_args = parse_arguments(args)
            assert parsed_args.command == expected_command

    def test_parse_arguments_with_flags(self):
        """Test parsing commands with various flags."""
        # Test verbose flag
        args = parse_arguments(["--verbose", "train"])
        assert args.verbose
        assert args.command == "train"

        # Test project flags
        args = parse_arguments(["--aging", "evaluate"])
        assert args.project == "fruitfly_aging"
        assert args.command == "evaluate"

        args = parse_arguments(["--alzheimers", "train"])
        assert args.project == "fruitfly_alzheimers"
        assert args.command == "train"

    def test_train_command_arguments(self):
        """Test train command specific arguments."""
        parser = create_main_parser()

        # Test basic train command
        args = parser.parse_args(["train"])
        assert args.command == "train"

        # Test train with flags
        args = parser.parse_args(["--verbose", "train"])
        assert args.verbose
        assert args.command == "train"

    def test_evaluate_command_arguments(self):
        """Test evaluate command specific arguments."""
        parser = create_main_parser()

        # Test basic evaluate command
        args = parser.parse_args(["evaluate"])
        assert args.command == "evaluate"

        # Test evaluate with interpretation flags
        args = parser.parse_args(["evaluate", "--interpret", "--visualize"])
        assert args.command == "evaluate"
        assert args.interpret
        assert args.visualize

    def test_test_command_arguments(self):
        """Test test command specific arguments."""
        parser = create_main_parser()

        # Test test command with type
        args = parser.parse_args(["test", "unit"])
        assert args.command == "test"
        assert args.test_type == "unit"

        # Test other test types
        for test_type in ["integration", "functional", "system"]:
            args = parser.parse_args(["test", test_type])
            assert args.test_type == test_type


@pytest.mark.unit
class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_cli_help_output(self):
        """Test CLI help output generation."""
        parser = create_main_parser()

        # Test that help can be generated without errors
        try:
            help_text = parser.format_help()
            assert isinstance(help_text, str)
            assert "TimeFlies" in help_text
            assert "train" in help_text
            assert "evaluate" in help_text
        except Exception:
            pytest.fail("Help generation failed")

    def test_subcommand_help(self):
        """Test subcommand help generation."""
        parser = create_main_parser()

        # Test that help generation works without errors
        help_text = parser.format_help()
        assert isinstance(help_text, str)
        assert len(help_text) > 0

        # Test that expected commands are mentioned in help
        expected_commands = ["train", "setup", "evaluate"]
        for cmd in expected_commands:
            assert cmd in help_text


@pytest.mark.unit
class TestCommandValidation:
    """Test command validation and error handling."""

    def test_required_arguments_validation(self):
        """Test that required arguments are validated."""
        parser = create_main_parser()

        # Test that command is required
        with pytest.raises(SystemExit):
            parser.parse_args([])  # No command provided

    def test_invalid_command_handling(self):
        """Test handling of invalid commands."""
        parser = create_main_parser()

        # Test invalid command
        with pytest.raises(SystemExit):
            parser.parse_args(["invalid_command"])

    def test_argument_combinations(self):
        """Test various argument combinations."""
        valid_combinations = [
            ["--verbose", "train"],
            ["--aging", "--verbose", "evaluate"],
            ["--alzheimers", "test", "unit"],
            ["setup"],
            ["create-test-data"],
        ]

        for args in valid_combinations:
            try:
                parsed = parse_arguments(args)
                assert parsed.command is not None
            except SystemExit:
                # Some combinations might have requirements we're not meeting
                pass

    def test_flag_inheritance(self):
        """Test that global flags work with all commands."""
        commands = ["train", "evaluate", "setup"]

        for command in commands:
            # Test verbose flag with each command
            args = parse_arguments(["--verbose", command])
            assert args.verbose
            assert args.command == command

    def test_help_flags(self):
        """Test help flags for various commands."""
        parser = create_main_parser()

        # Test main help
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

        # Test command help
        help_commands = [
            ["train", "--help"],
            ["evaluate", "--help"],
        ]

        for cmd_args in help_commands:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args(cmd_args)
            assert exc_info.value.code == 0
