"""Basic component tests moved from functional."""

import pytest
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.core.config_manager import Config
from shared.data.loaders import DataLoader
from shared.cli.parser import create_main_parser, parse_arguments


def test_config_creation():
    """Test basic config creation and access."""
    config_dict = {
        'general': {'project_name': 'TimeFlies', 'version': '0.2.0', 'random_state': 42},
        'data': {
            'tissue': 'head',
            'encoding_variable': 'age',
            'model_type': 'CNN',
            'batch_correction': {'enabled': False}
        }
    }
    
    config = Config(config_dict)
    assert config.general.project_name == 'TimeFlies'
    assert config.data.tissue == 'head'
    assert config.data.encoding_variable == 'age'


def test_cli_parser_commands():
    """Test CLI parser with different commands."""
    parser = create_main_parser()
    
    # Test train command
    args = parser.parse_args(['train', '--tissue', 'head', '--model', 'cnn'])
    assert args.command == 'train'
    assert args.tissue == 'head'
    assert args.model == 'cnn'
    
    # Test setup command
    args = parser.parse_args(['setup'])
    assert args.command == 'setup'
    
    # Test evaluate command
    args = parser.parse_args(['evaluate', '--interpret', '--visualize'])
    assert args.command == 'evaluate'
    assert args.interpret == True
    assert args.visualize == True


def test_parse_arguments_function():
    """Test parse_arguments function."""
    args = parse_arguments(['train', '--model', 'mlp'])
    assert args.command == 'train'
    assert args.model == 'mlp'


def test_data_loader_creation():
    """Test data loader creation."""
    mock_config = Mock()
    mock_config.file_locations.training_file = 'test_train.h5ad'
    mock_config.file_locations.evaluation_file = 'test_eval.h5ad'
    mock_config.file_locations.original_file = 'test_original.h5ad'
    mock_config.file_locations.batch_corrected_files.train = 'test_train_batch.h5ad'
    mock_config.file_locations.batch_corrected_files.eval = 'test_eval_batch.h5ad'
    
    with patch('shared.data.loaders.PathManager'):
        loader = DataLoader(mock_config)
        assert loader.config == mock_config


def test_exceptions_creation():
    """Test custom exceptions."""
    from shared.utils.exceptions import ModelError, DataError, ConfigurationError
    
    # Test ModelError
    with pytest.raises(ModelError):
        raise ModelError("Test model error")
    
    # Test DataError  
    with pytest.raises(DataError):
        raise DataError("Test data error")
    
    # Test ConfigurationError
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Test config error")