#!/usr/bin/env python3
"""
Modern TimeFlies Test Suite

Comprehensive tests for the refactored TimeFlies codebase.
Tests the current architecture, CLI design, and config structure.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.cli.commands import evaluate_command, run_system_tests, train_command
from common.cli.parser import create_main_parser, parse_arguments
from common.core.active_config import get_active_project, get_config_for_active_project


class TestCLICommands:
    """Test CLI command functionality."""

    def test_verify_command_runs(self):
        """Test that verify command executes without errors."""
        # Mock args and config
        args = Mock()
        args.verbose = False
        args.project = None

        # This should run the system verification
        # Note: May fail on missing data, but should not crash
        try:
            result = run_system_tests(args)
            assert isinstance(result, int)  # Should return exit code
        except Exception as e:
            # Acceptable if it fails due to missing data, not code errors
            assert "not found" in str(e) or "missing" in str(e).lower()

    def test_cli_parser_basic(self):
        """Test basic CLI parser functionality."""
        parser = create_main_parser()

        # Test basic commands
        args = parser.parse_args(['verify'])
        assert args.command == 'verify'

        args = parser.parse_args(['train'])
        assert args.command == 'train'

        args = parser.parse_args(['evaluate'])
        assert args.command == 'evaluate'

    def test_cli_project_flags(self):
        """Test project switching flags."""
        parser = create_main_parser()

        # Test aging flag
        args = parser.parse_args(['--aging', 'train'])
        assert args.project == 'fruitfly_aging'

        # Test alzheimers flag
        args = parser.parse_args(['--alzheimers', 'evaluate'])
        assert args.project == 'fruitfly_alzheimers'

        # Test no flag (should be None)
        args = parser.parse_args(['train'])
        assert args.project is None

    def test_all_cli_commands_parseable(self):
        """Test that all CLI commands can be parsed without errors."""
        parser = create_main_parser()

        # Test all basic commands
        commands_to_test = [
            ['verify'],
            ['setup'],
            ['train'],
            ['evaluate'],
            ['test', 'unit'],
            ['batch-correct'],
            ['create-test-data']
        ]

        for cmd_args in commands_to_test:
            try:
                args = parser.parse_args(cmd_args)
                assert args.command is not None
            except SystemExit:
                # Some commands might have required args, that's ok for parsing test
                pass

    def test_cli_help_commands(self):
        """Test that help works for all commands."""
        parser = create_main_parser()

        # Test main help
        try:
            parser.parse_args(['--help'])
        except SystemExit:
            pass  # Help exits with 0, which is expected

        # Test command-specific help
        help_commands = [
            ['train', '--help'],
            ['evaluate', '--help'],
            ['verify', '--help']
        ]

        for cmd_args in help_commands:
            try:
                parser.parse_args(cmd_args)
            except SystemExit:
                pass  # Help exits, which is expected


class TestConfigSystem:
    """Test configuration system."""

    def test_active_project_detection(self):
        """Test active project detection from config."""
        project = get_active_project()
        assert project in ['fruitfly_aging', 'fruitfly_alzheimers']

    def test_config_loading(self):
        """Test config loading for both projects."""
        # Test aging project
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        assert hasattr(config, 'data')
        assert hasattr(config, 'general')
        assert hasattr(config.data, 'tissue')
        assert hasattr(config.data, 'model')
        assert hasattr(config.data, 'target_variable')

    def test_config_structure(self):
        """Test modern config structure."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Test modern structure (not legacy)
        assert hasattr(config.data, 'split')  # New structure
        assert hasattr(config.data.split, 'method')
        assert hasattr(config.data.split, 'test_ratio')
        assert hasattr(config, 'hardware')  # Not 'device'
        assert hasattr(config.hardware, 'processor')


class TestDataSampling:
    """Test data sampling functionality."""

    def test_sampling_config_structure(self):
        """Test sampling configuration structure."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        assert hasattr(config.data, 'sampling')
        assert hasattr(config.data.sampling, 'samples')  # Not 'num_samples'
        assert hasattr(config.data.sampling, 'variables')  # Not 'num_variables'


class TestSplitMethods:
    """Test different data splitting methods."""

    def test_split_config_options(self):
        """Test split method configuration options."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Test split methods exist
        assert hasattr(config.data.split, 'method')
        method = config.data.split.method
        assert method in ['random', 'sex', 'tissue']

        # Test sex split options
        if hasattr(config.data.split, 'sex'):
            assert hasattr(config.data.split.sex, 'train')
            assert hasattr(config.data.split.sex, 'test')
            assert hasattr(config.data.split.sex, 'test_ratio')

        # Test tissue split options
        if hasattr(config.data.split, 'tissue'):
            assert hasattr(config.data.split.tissue, 'train')
            assert hasattr(config.data.split.tissue, 'test')
            assert hasattr(config.data.split.tissue, 'test_ratio')


class TestModelTypes:
    """Test different model types."""

    def test_model_config(self):
        """Test model configuration."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        assert hasattr(config.data, 'model')
        model_type = config.data.model.lower()
        assert model_type in ['cnn', 'mlp']


class TestOptionalFeatures:
    """Test optional features like EDA and SHAP."""

    def test_eda_config(self):
        """Test EDA configuration structure."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # EDA can be in analysis section
        if hasattr(config, 'analysis') and hasattr(config.analysis, 'eda'):
            assert hasattr(config.analysis.eda, 'enabled')

    def test_shap_config(self):
        """Test SHAP configuration structure."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # SHAP should be in interpretation.shap (modern structure)
        assert hasattr(config, 'interpretation')
        assert hasattr(config.interpretation, 'shap')
        assert hasattr(config.interpretation.shap, 'enabled')
        assert hasattr(config.interpretation.shap, 'reference_size')

    def test_visualization_config(self):
        """Test visualization configuration."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        assert hasattr(config, 'visualizations')
        assert hasattr(config.visualizations, 'enabled')


class TestEDAAndInterpretation:
    """Test EDA and interpretation features."""

    def test_eda_functionality_config(self):
        """Test EDA can be enabled and configured."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Test EDA config structure
        assert hasattr(config, 'analysis')
        assert hasattr(config.analysis, 'eda')
        assert hasattr(config.analysis.eda, 'enabled')

        # Test legacy structure also exists
        assert hasattr(config, 'data_processing')
        assert hasattr(config.data_processing, 'exploratory_data_analysis')

    def test_shap_functionality_config(self):
        """Test SHAP can be enabled and configured."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Test SHAP config structure
        assert hasattr(config.interpretation.shap, 'enabled')
        assert hasattr(config.interpretation.shap, 'reference_size')
        assert hasattr(config.interpretation.shap, 'load_existing')

        # Test values are sensible
        assert isinstance(config.interpretation.shap.enabled, bool)
        assert isinstance(config.interpretation.shap.reference_size, int)
        assert config.interpretation.shap.reference_size > 0


class TestAdvancedSplitMethods:
    """Test advanced data splitting methods."""

    def test_random_split_config(self):
        """Test random split configuration."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # When method is random, should use test_ratio
        if config.data.split.method == 'random':
            assert hasattr(config.data.split, 'test_ratio')
            assert 0 < config.data.split.test_ratio < 1

    def test_sex_split_config(self):
        """Test sex-based split configuration."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Test sex split has required attributes
        assert hasattr(config.data.split, 'sex')
        assert hasattr(config.data.split.sex, 'train')
        assert hasattr(config.data.split.sex, 'test')
        assert hasattr(config.data.split.sex, 'test_ratio')

        # Test values are valid
        valid_sexes = ['male', 'female', 'all']
        assert config.data.split.sex.train in valid_sexes
        assert config.data.split.sex.test in valid_sexes
        assert 0 < config.data.split.sex.test_ratio < 1

    def test_tissue_split_config(self):
        """Test tissue-based split configuration."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Test tissue split has required attributes
        assert hasattr(config.data.split, 'tissue')
        assert hasattr(config.data.split.tissue, 'train')
        assert hasattr(config.data.split.tissue, 'test')
        assert hasattr(config.data.split.tissue, 'test_ratio')

        # Test values are valid
        valid_tissues = ['head', 'body', 'all']
        assert config.data.split.tissue.train in valid_tissues
        assert config.data.split.tissue.test in valid_tissues
        assert 0 < config.data.split.tissue.test_ratio < 1


class TestBatchCorrection:
    """Test batch correction functionality."""

    def test_batch_correction_config(self):
        """Test batch correction configuration."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        assert hasattr(config.data, 'batch_correction')
        assert hasattr(config.data.batch_correction, 'enabled')
        assert isinstance(config.data.batch_correction.enabled, bool)

    def test_batch_file_paths(self):
        """Test batch-corrected file paths exist in config."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Check batch file paths are defined
        assert hasattr(config, 'paths')
        assert hasattr(config.paths, 'batch_data')
        assert hasattr(config.paths.batch_data, 'train')
        assert hasattr(config.paths.batch_data, 'eval')

        # Paths should contain expected tokens
        assert '{project}' in config.paths.batch_data.train
        assert '{tissue}' in config.paths.batch_data.train
        assert 'batch' in config.paths.batch_data.train


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_config_graceful(self):
        """Test graceful handling of missing config sections."""
        config_manager = get_config_for_active_project('default')
        config = config_manager.get_config()

        # Should not crash when accessing optional sections
        batch_enabled = getattr(config.data.batch_correction, 'enabled', False)
        assert isinstance(batch_enabled, bool)

        random_state = getattr(config.general, 'random_state', 42)
        assert isinstance(random_state, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
