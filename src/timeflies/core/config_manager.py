"""Modern configuration management with YAML support."""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from ..utils.exceptions import ConfigurationError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class Config:
    """Modern configuration class with nested attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._data = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                self._data[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to nested configurations."""
        if name.startswith('_'):
            raise AttributeError(f"'{name}' not found")
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"Configuration '{name}' not found")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting configuration values."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__(name, value)
            else:
                if isinstance(value, dict):
                    value = Config(value)
                self._data[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._data.get(key, default)
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in other.items():
            if isinstance(value, dict) and key in self._data and isinstance(self._data[key], Config):
                self._data[key].update(value)
            else:
                if isinstance(value, dict):
                    value = Config(value)
                self._data[key] = value
    
    def __repr__(self) -> str:
        return f"Config({self._data})"


class ConfigManager:
    """Modern configuration manager with YAML support and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = self._find_config_path(config_path)
        self._config = None
        self._load_config()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create a Config object directly from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration data
            
        Returns:
            Config object initialized with the provided dictionary
        """
        return Config(config_dict)
    
    def _find_config_path(self, config_path: Optional[str]) -> str:
        """Find configuration file path."""
        if config_path:
            if os.path.exists(config_path):
                return config_path
            else:
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
        # Try to find default config file
        default_paths = [
            "configs/default.yaml",
            "../configs/default.yaml", 
            "../../configs/default.yaml",
            os.path.join(os.path.dirname(__file__), "../../configs/default.yaml")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                return path
                
        raise ConfigurationError("No configuration file found. Please provide a valid config path.")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create Config object directly from YAML structure
            self._config = Config(config_dict)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {self.config_path}: {e}")
    
    def validate_config(self) -> None:
        """Validate configuration for required sections and parameters.""" 
        required_sections = ["data", "model", "general"]
        
        for section in required_sections:
            if not hasattr(self._config, section):
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate model type
        from ..utils.constants import SUPPORTED_MODEL_TYPES
        model_type = self._config.data.model_type.lower()
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ConfigurationError(f"Invalid model type: {model_type}")
        
        # Validate encoding variable
        from ..utils.constants import SUPPORTED_ENCODING_VARS
        encoding_var = self._config.data.encoding_variable.lower()
        if encoding_var not in SUPPORTED_ENCODING_VARS:
            raise ConfigurationError(f"Invalid encoding variable: {encoding_var}")
            
        logger.info("Configuration validation passed")
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file."""
        config_dict = self._config.to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {output_path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(updates)
        logger.info("Configuration updated")
    
    def get_config(self) -> Config:
        """Get the configuration object."""
        return self._config


# Global configuration instance  
_config_manager = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        Configuration object
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
        _config_manager.validate_config()
    
    return _config_manager.get_config()


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
        _config_manager.validate_config()
    
    return _config_manager


def reset_config() -> None:
    """Reset global configuration instance."""
    global _config_manager
    _config_manager = None