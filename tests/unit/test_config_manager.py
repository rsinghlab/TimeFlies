"""Tests for configuration management."""

import pytest
import tempfile
import os
import yaml
from pathlib import Path

from src.timeflies.core.config_manager import Config, ConfigManager, get_config, reset_config
from src.timeflies.utils.exceptions import ConfigurationError


class TestConfig:
    """Test Config class functionality."""
    
    def test_config_initialization(self):
        """Test Config initialization with nested dictionaries."""
        config_dict = {
            "general": {"project_name": "test", "version": "1.0"},
            "data": {"tissue": "head", "model_type": "CNN"}
        }
        config = Config(config_dict)
        
        assert config.general.project_name == "test"
        assert config.data.tissue == "head"
        
    def test_config_attribute_access(self):
        """Test attribute access for nested configs."""
        config_dict = {"level1": {"level2": {"value": "test"}}}
        config = Config(config_dict)
        
        assert config.level1.level2.value == "test"
        
    def test_config_missing_attribute(self):
        """Test error handling for missing attributes."""
        config = Config({"existing": "value"})
        
        with pytest.raises(AttributeError):
            _ = config.missing_attribute
            
    def test_config_to_dict(self):
        """Test conversion back to dictionary."""
        original_dict = {"a": {"b": "value"}, "c": "another"}
        config = Config(original_dict)
        result_dict = config.to_dict()
        
        assert result_dict == original_dict
        
    def test_config_update(self):
        """Test config update functionality."""
        config = Config({"a": {"b": "old"}, "c": "value"})
        config.update({"a": {"b": "new", "d": "added"}})
        
        assert config.a.b == "new"
        assert config.a.d == "added"
        assert config.c == "value"


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
        
    def test_config_manager_with_yaml(self):
        """Test ConfigManager with YAML file."""
        config_data = {
            "general": {"project_name": "test"},
            "data": {"tissue": "head", "model_type": "CNN"},
            "model": {"training": {"epochs": 100}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            manager = ConfigManager(config_path)
            config = manager.get_config()
            
            assert config.general.project_name == "test"
            assert config.data.tissue == "head"
            assert config.model.training.epochs == 100
        finally:
            os.unlink(config_path)
            
    def test_config_manager_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(ConfigurationError):
            ConfigManager("/nonexistent/config.yaml")
            
    def test_config_validation_success(self):
        """Test successful config validation."""
        config_data = {
            "general": {"project_name": "test"},
            "data": {"model_type": "cnn", "encoding_variable": "age"},
            "model": {"training": {"epochs": 100}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            manager = ConfigManager(config_path)
            manager.validate_config()  # Should not raise
        finally:
            os.unlink(config_path)
            
    def test_config_validation_invalid_model(self):
        """Test validation error for invalid model type."""
        config_data = {
            "general": {"project_name": "test"},
            "data": {"model_type": "invalid_model", "encoding_variable": "age"},
            "model": {"training": {"epochs": 100}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            manager = ConfigManager(config_path)
            with pytest.raises(ConfigurationError):
                manager.validate_config()
        finally:
            os.unlink(config_path)
            
    def test_save_config(self):
        """Test saving config to file."""
        config_data = {
            "general": {"project_name": "test"},
            "data": {"tissue": "head"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
            
        try:
            manager = ConfigManager(config_path)
            manager.save_config(output_path)
            
            # Verify file was created and contains expected data
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            assert saved_data["general"]["project_name"] == "test"
        finally:
            os.unlink(config_path)
            os.unlink(output_path)


class TestGlobalConfig:
    """Test global configuration functions."""
    
    def setup_method(self):
        """Reset global config before each test."""
        reset_config()
        
    def test_get_config_creates_singleton(self):
        """Test that get_config creates singleton instance."""
        config_data = {
            "general": {"project_name": "test"},
            "data": {"model_type": "cnn", "encoding_variable": "age"},
            "model": {"training": {"epochs": 100}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            config1 = get_config(config_path)
            config2 = get_config()  # Should return same instance
            
            assert config1.general.project_name == config2.general.project_name
        finally:
            os.unlink(config_path)
            
    def test_reset_config(self):
        """Test config reset functionality."""
        config_data = {
            "general": {"project_name": "test"},
            "data": {"model_type": "cnn", "encoding_variable": "age"},
            "model": {"training": {"epochs": 100}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            config1 = get_config(config_path)
            reset_config()
            config2 = get_config(config_path)
            
            # Should be new instance but same values
            assert config1.general.project_name == config2.general.project_name
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])