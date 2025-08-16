"""Tests for utility modules."""

import pytest
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from timeflies.utils.logging_config import setup_logging, get_logger, LoggerMixin
from timeflies.utils.exceptions import (
    TimeFliesError, ConfigurationError, DataError, ModelError
)
from timeflies.utils.constants import (
    SUPPORTED_MODEL_TYPES, DEFAULT_N_TOP_GENES, ERROR_MESSAGES
)
from timeflies.utils.path_manager import PathManager


class TestLoggingConfig:
    """Test logging configuration utilities."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        setup_logging(level="DEBUG")
        logger = get_logger("test")
        assert logger.level <= logging.DEBUG
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            setup_logging(level="INFO", log_file=log_file)
            
            logger = get_logger("test")
            logger.info("Test message")
            
            assert os.path.exists(log_file)
    
    def test_logger_mixin(self):
        """Test LoggerMixin functionality."""
        class TestClass(LoggerMixin):
            def __init__(self):
                super().__init__()
        
        test_obj = TestClass()
        assert hasattr(test_obj, 'logger')
        assert isinstance(test_obj.logger, logging.Logger)


class TestExceptions:
    """Test custom exception classes."""
    
    def test_timeflies_error_basic(self):
        """Test basic TimeFliesError."""
        error = TimeFliesError("Test error")
        assert str(error) == "Test error"
    
    def test_timeflies_error_with_details(self):
        """Test TimeFliesError with details."""
        details = {"key": "value"}
        error = TimeFliesError("Test error", details=details)
        assert "Details:" in str(error)
        assert error.details == details
    
    def test_configuration_error(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Config error")
        assert isinstance(error, TimeFliesError)
    
    def test_data_error(self):
        """Test DataError inheritance."""
        error = DataError("Data error")
        assert isinstance(error, TimeFliesError)
    
    def test_model_error(self):
        """Test ModelError inheritance."""
        error = ModelError("Model error")
        assert isinstance(error, TimeFliesError)


class TestConstants:
    """Test constants and configuration values."""
    
    def test_supported_model_types(self):
        """Test that supported model types are defined."""
        assert isinstance(SUPPORTED_MODEL_TYPES, list)
        assert "cnn" in SUPPORTED_MODEL_TYPES
        assert "mlp" in SUPPORTED_MODEL_TYPES
    
    def test_default_values(self):
        """Test default configuration values."""
        assert DEFAULT_N_TOP_GENES == 5000
        assert isinstance(ERROR_MESSAGES, dict)
    
    def test_error_messages(self):
        """Test error message templates."""
        assert "config_missing_section" in ERROR_MESSAGES
        assert "invalid_model_type" in ERROR_MESSAGES


class TestPathManager:
    """Test path management utilities."""
    
    def setUp(self):
        """Set up mock configuration."""
        self.mock_config = MagicMock()
        self.mock_config.DataParameters.BatchCorrection.enabled = False
        self.mock_config.DataParameters.GeneralSettings.tissue = "head"
        self.mock_config.DataParameters.GeneralSettings.model_type = "CNN"
        self.mock_config.DataParameters.GeneralSettings.encoding_variable = "age"
        self.mock_config.DataParameters.GeneralSettings.cell_type = "all"
        self.mock_config.DataParameters.GeneralSettings.sex_type = "all"
        self.mock_config.DataParameters.Sampling.num_samples = None
        self.mock_config.GenePreprocessing.GeneFiltering.highly_variable_genes = False
        self.mock_config.GenePreprocessing.GeneBalancing.balance_genes = False
        self.mock_config.GenePreprocessing.GeneBalancing.balance_lnc_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.select_batch_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.only_keep_lnc_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.remove_lnc_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.remove_autosomal_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.remove_sex_genes = False
        self.mock_config.DataParameters.TrainTestSplit.method = "random"
    
    def test_path_manager_initialization(self):
        """Test PathManager initialization."""
        self.setUp()
        path_manager = PathManager(self.mock_config)
        
        assert path_manager.correction_dir == "uncorrected"
        assert path_manager.tissue == "head"
        assert path_manager.model_type == "CNN"
    
    def test_construct_model_directory(self):
        """Test model directory construction."""
        self.setUp()
        path_manager = PathManager(self.mock_config)
        model_dir = path_manager.construct_model_directory()
        
        assert isinstance(model_dir, str)
        assert "Models" in model_dir
        assert "uncorrected" in model_dir
        assert "head" in model_dir
    
    def test_get_visualization_directory(self):
        """Test visualization directory construction."""
        self.setUp()
        path_manager = PathManager(self.mock_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('os.path.dirname') as mock_dirname:
                mock_dirname.return_value = tmpdir
                viz_dir = path_manager.get_visualization_directory()
                
                assert isinstance(viz_dir, str)
                assert "Results" in viz_dir


if __name__ == "__main__":
    pytest.main([__file__])