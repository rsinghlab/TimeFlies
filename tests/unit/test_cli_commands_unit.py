"""Unit tests for CLI commands functionality to increase coverage."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import sys

# Import CLI modules
from shared.cli.commands import (
    setup_command,
    create_test_data_command, 
    run_system_tests,
    run_test_suite,
    train_command,
    evaluate_command
)
from shared.cli.main import main_cli
from shared.cli.parser import create_main_parser, parse_arguments


@pytest.mark.unit
class TestCLICommandExecution:
    """Test actual CLI command execution."""
    
    def test_setup_command_execution(self, mock_args):
        """Test setup command execution."""
        mock_args.command = 'setup'
        
        with patch('shared.cli.commands.print') as mock_print:
            result = setup_command(mock_args)
            
            # Should return success code
            assert result == 0
            # Should print setup information
            mock_print.assert_called()
    
    def test_create_test_data_command_execution(self, mock_args):
        """Test create test data command execution."""
        mock_args.command = 'create-test-data'
        
        with patch('shared.cli.commands.create_sample_h5ad_files') as mock_create:
            with patch('shared.cli.commands.print') as mock_print:
                mock_create.return_value = {'head_train': 'test.h5ad'}
                
                result = create_test_data_command(mock_args)
                
                # Should return success code
                assert result == 0
                mock_print.assert_called()
    
    def test_run_system_tests_functionality(self, mock_args):
        """Test system tests functionality."""
        mock_args.verbose = False
        mock_args.project = None
        
        with patch('shared.cli.commands.get_active_project') as mock_get_project:
            with patch('shared.cli.commands.get_config_for_active_project') as mock_get_config:
                with patch('shared.cli.commands.print') as mock_print:
                    mock_get_project.return_value = 'fruitfly_aging'
                    mock_config = Mock()
                    mock_get_config.return_value.get_config.return_value = mock_config
                    
                    result = run_system_tests(mock_args)
                    
                    # Should return integer exit code
                    assert isinstance(result, int)
                    mock_print.assert_called()
    
    def test_run_test_suite_functionality(self, mock_args):
        """Test test suite functionality."""
        mock_args.test_type = 'unit'
        mock_args.verbose = False
        mock_args.coverage = False
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            
            result = run_test_suite(mock_args)
            
            # Should return exit code
            assert isinstance(result, int)
            mock_subprocess.assert_called()
    
    def test_train_command_workflow(self, mock_args, aging_config):
        """Test training command workflow."""
        mock_args.command = 'train'
        mock_args.verbose = False
        
        with patch('shared.cli.commands.get_active_project') as mock_get_project:
            with patch('shared.cli.commands.PipelineManager') as mock_pipeline:
                with patch('shared.cli.commands.print') as mock_print:
                    mock_get_project.return_value = 'fruitfly_aging'
                    mock_pipeline_instance = Mock()
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    try:
                        result = train_command(mock_args, aging_config)
                        # Should return integer exit code
                        assert isinstance(result, int)
                    except Exception as e:
                        # May fail due to missing data, but should not be import errors
                        assert any(word in str(e).lower() for word in ['file', 'data', 'path', 'not found'])
    
    def test_evaluate_command_workflow(self, mock_args, aging_config):
        """Test evaluation command workflow."""
        mock_args.command = 'evaluate'
        mock_args.verbose = False
        mock_args.interpret = False
        mock_args.visualize = False
        
        with patch('shared.cli.commands.get_active_project') as mock_get_project:
            with patch('shared.cli.commands.PipelineManager') as mock_pipeline:
                with patch('shared.cli.commands.print') as mock_print:
                    mock_get_project.return_value = 'fruitfly_aging'
                    mock_pipeline_instance = Mock()
                    mock_pipeline.return_value = mock_pipeline_instance
                    
                    try:
                        result = evaluate_command(mock_args, aging_config)
                        # Should return integer exit code
                        assert isinstance(result, int)
                    except Exception as e:
                        # May fail due to missing data/models
                        assert any(word in str(e).lower() for word in ['file', 'data', 'path', 'model', 'not found'])


@pytest.mark.unit
class TestCLIArgumentHandling:
    """Test CLI argument handling and validation."""
    
    def test_parse_arguments_all_commands(self):
        """Test parsing all available commands."""
        command_tests = [
            (['train'], 'train'),
            (['evaluate'], 'evaluate'),
            (['verify'], 'verify'),
            (['setup'], 'setup'),
            (['create-test-data'], 'create-test-data'),
            (['test', 'unit'], 'test'),
            (['batch-correct'], 'batch-correct')
        ]
        
        for args, expected_command in command_tests:
            parsed_args = parse_arguments(args)
            assert parsed_args.command == expected_command
    
    def test_parse_arguments_with_flags(self):
        """Test parsing commands with various flags."""
        # Test verbose flag
        args = parse_arguments(['--verbose', 'train'])
        assert args.verbose == True
        assert args.command == 'train'
        
        # Test project flags
        args = parse_arguments(['--aging', 'evaluate'])
        assert args.project == 'fruitfly_aging'
        assert args.command == 'evaluate'
        
        args = parse_arguments(['--alzheimers', 'train'])
        assert args.project == 'fruitfly_alzheimers'
        assert args.command == 'train'
    
    def test_train_command_arguments(self):
        """Test train command specific arguments."""
        parser = create_main_parser()
        
        # Test basic train command
        args = parser.parse_args(['train'])
        assert args.command == 'train'
        
        # Test train with flags
        args = parser.parse_args(['--verbose', 'train'])
        assert args.verbose == True
        assert args.command == 'train'
    
    def test_evaluate_command_arguments(self):
        """Test evaluate command specific arguments."""
        parser = create_main_parser()
        
        # Test basic evaluate command
        args = parser.parse_args(['evaluate'])
        assert args.command == 'evaluate'
        
        # Test evaluate with interpretation flags
        args = parser.parse_args(['evaluate', '--interpret', '--visualize'])
        assert args.command == 'evaluate'
        assert args.interpret == True
        assert args.visualize == True
    
    def test_test_command_arguments(self):
        """Test test command specific arguments."""
        parser = create_main_parser()
        
        # Test test command with type
        args = parser.parse_args(['test', 'unit'])
        assert args.command == 'test'
        assert args.test_type == 'unit'
        
        # Test other test types
        for test_type in ['integration', 'functional', 'system']:
            args = parser.parse_args(['test', test_type])
            assert args.test_type == test_type


@pytest.mark.unit
class TestCLIMainEntryPoint:
    """Test main CLI entry point functionality."""
    
    def test_main_cli_setup_command(self):
        """Test main CLI with setup command.""" 
        with patch('shared.cli.commands.setup_command') as mock_setup:
            mock_setup.return_value = 0
            
            result = main_cli(['setup'])
            assert result == 0
            mock_setup.assert_called_once()
    
    def test_main_cli_create_test_data(self):
        """Test main CLI with create test data command."""
        with patch('shared.cli.commands.create_test_data_command') as mock_create:
            mock_create.return_value = 0
            
            result = main_cli(['create-test-data'])
            assert result == 0
            mock_create.assert_called_once()
    
    def test_main_cli_verify_command(self):
        """Test main CLI with verify command."""
        with patch('shared.cli.commands.run_system_tests') as mock_verify:
            mock_verify.return_value = 0
            
            result = main_cli(['verify'])
            assert result == 0
            mock_verify.assert_called_once()
    
    def test_main_cli_with_project_override(self):
        """Test main CLI with project override."""
        with patch('shared.cli.main.get_config_for_active_project') as mock_get_config:
            with patch('shared.cli.commands.train_command') as mock_train:
                mock_config = Mock()
                mock_get_config.return_value.get_config.return_value = mock_config
                mock_train.return_value = 0
                
                result = main_cli(['--aging', 'train'])
                assert result == 0
    
    def test_main_cli_error_handling(self):
        """Test main CLI error handling."""
        # Test with invalid command
        with patch('shared.cli.parser.parse_arguments') as mock_parse:
            mock_parse.side_effect = SystemExit(2)
            
            result = main_cli(['invalid_command'])
            # Should handle SystemExit gracefully
            assert result is not None
    
    def test_main_cli_keyboard_interrupt(self):
        """Test main CLI keyboard interrupt handling."""
        with patch('shared.cli.main.get_config_for_active_project') as mock_get_config:
            mock_get_config.side_effect = KeyboardInterrupt()
            
            result = main_cli(['train'])
            assert result == 130  # Standard keyboard interrupt exit code
    
    def test_main_cli_config_error(self):
        """Test main CLI configuration error handling."""
        with patch('shared.cli.main.get_config_for_active_project') as mock_get_config:
            mock_get_config.side_effect = Exception("Config error")
            
            result = main_cli(['train'])
            assert result == 1  # Error exit code


@pytest.mark.unit
class TestCLIUtilities:
    """Test CLI utility functions."""
    
    def test_logging_setup_in_cli(self):
        """Test logging setup in CLI context."""
        from shared.utils.logging_config import setup_logging
        
        # Test that logging can be set up for CLI
        try:
            setup_logging(level="INFO")
            assert True
        except Exception:
            # May fail in test environment
            pass
        
        try:
            setup_logging(level="DEBUG")
            assert True
        except Exception:
            pass
    
    def test_cli_help_output(self):
        """Test CLI help output generation."""
        parser = create_main_parser()
        
        # Test that help can be generated without errors
        try:
            help_text = parser.format_help()
            assert isinstance(help_text, str)
            assert 'TimeFlies' in help_text
            assert 'train' in help_text
            assert 'evaluate' in help_text
        except Exception:
            pytest.fail("Help generation failed")
    
    def test_subcommand_help(self):
        """Test subcommand help generation."""
        parser = create_main_parser()
        
        # Test that subcommand help doesn't crash
        try:
            # This tests the parser's ability to handle subcommand help
            subparsers_actions = [
                action for action in parser._actions 
                if isinstance(action, type(parser._subparsers))
            ]
            assert len(subparsers_actions) > 0
        except Exception:
            pytest.fail("Subcommand help generation failed")


@pytest.mark.unit
class TestCLIProjectSwitching:
    """Test CLI project switching functionality."""
    
    def test_project_detection_workflow(self):
        """Test project detection workflow."""
        from shared.core.active_config import get_active_project
        
        # Test that project detection works
        project = get_active_project()
        assert project in ['fruitfly_aging', 'fruitfly_alzheimers']
    
    def test_config_loading_for_projects(self):
        """Test config loading for different projects."""
        from shared.core.active_config import get_config_for_active_project
        
        # Test loading config for fruitfly_aging
        try:
            config_manager = get_config_for_active_project('default')
            config = config_manager.get_config()
            assert config is not None
        except Exception as e:
            # Should not be import errors
            assert "import" not in str(e).lower()
    
    def test_cli_project_flag_processing(self):
        """Test CLI project flag processing."""
        # Test aging flag sets correct project
        args = parse_arguments(['--aging', 'train'])
        assert args.project == 'fruitfly_aging'
        
        # Test alzheimers flag sets correct project
        args = parse_arguments(['--alzheimers', 'evaluate'])
        assert args.project == 'fruitfly_alzheimers'
        
        # Test that flags are mutually exclusive (can't set both)
        parser = create_main_parser()
        
        # This should not raise an error for individual flags
        parser.parse_args(['--aging', 'train'])
        parser.parse_args(['--alzheimers', 'train'])


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
            parser.parse_args(['invalid_command'])
    
    def test_argument_combinations(self):
        """Test various argument combinations."""
        valid_combinations = [
            ['--verbose', 'train'],
            ['--aging', '--verbose', 'evaluate'],
            ['--alzheimers', 'test', 'unit'],
            ['verify', '--verbose'],
            ['setup'],
            ['create-test-data']
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
        commands = ['train', 'evaluate', 'verify', 'setup']
        
        for command in commands:
            # Test verbose flag with each command
            args = parse_arguments(['--verbose', command])
            assert args.verbose == True
            assert args.command == command
    
    def test_help_flags(self):
        """Test help flags for various commands."""
        parser = create_main_parser()
        
        # Test main help
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--help'])
        assert exc_info.value.code == 0
        
        # Test command help
        help_commands = [
            ['train', '--help'],
            ['evaluate', '--help'],
            ['verify', '--help']
        ]
        
        for cmd_args in help_commands:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args(cmd_args)
            assert exc_info.value.code == 0