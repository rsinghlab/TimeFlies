"""Enhanced configuration management with YAML support."""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from ..utils.exceptions import ConfigurationError
from ..utils.constants import REQUIRED_CONFIG_SECTIONS
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Config:
    """Configuration data class with nested attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to nested configurations."""
        raise AttributeError(f"Configuration '{name}' not found")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class ConfigManager:
    """Enhanced configuration manager with YAML support and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file or fallback to Python config."""
        if self.config_path and os.path.exists(self.config_path):
            self._load_yaml_config()
        else:
            # Fallback to original Python config
            self._load_python_config()
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to nested Config object
            self._config = self._convert_to_legacy_format(config_dict)
            logger.info(f"Loaded YAML configuration from {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML config: {e}")
    
    def _load_python_config(self) -> None:
        """Load original Python configuration as fallback."""
        try:
            from .config import config as python_config
            self._config = python_config
            logger.info("Loaded Python configuration (fallback)")
        except ImportError as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _convert_to_legacy_format(self, yaml_config: Dict[str, Any]) -> Config:
        """
        Convert YAML config to legacy format for backward compatibility.
        
        Args:
            yaml_config: Configuration dictionary from YAML
            
        Returns:
            Config object in legacy format
        """
        # Map YAML structure to legacy structure
        legacy_config = {
            "Device": {
                "processor": yaml_config.get("device", {}).get("processor", "GPU")
            },
            "DataParameters": {
                "GeneralSettings": {
                    "tissue": yaml_config.get("data", {}).get("tissue", "head"),
                    "model_type": yaml_config.get("data", {}).get("model_type", "CNN"),
                    "encoding_variable": yaml_config.get("data", {}).get("encoding_variable", "age"),
                    "cell_type": yaml_config.get("data", {}).get("cell_type", "all"),
                    "sex_type": yaml_config.get("data", {}).get("sex_type", "all")
                },
                "BatchCorrection": {
                    "enabled": yaml_config.get("data", {}).get("batch_correction", {}).get("enabled", True)
                },
                "Filtering": {
                    "include_mixed_sex": yaml_config.get("data", {}).get("filtering", {}).get("include_mixed_sex", False)
                },
                "Sampling": {
                    "num_samples": yaml_config.get("data", {}).get("sampling", {}).get("num_samples"),
                    "num_variables": yaml_config.get("data", {}).get("sampling", {}).get("num_variables")
                },
                "TrainTestSplit": {
                    "method": yaml_config.get("data", {}).get("train_test_split", {}).get("method", "random"),
                    "train": yaml_config.get("data", {}).get("train_test_split", {}).get("train", {}),
                    "test": yaml_config.get("data", {}).get("train_test_split", {}).get("test", {})
                }
            },
            "GenePreprocessing": {
                "GeneFiltering": yaml_config.get("gene_preprocessing", {}).get("gene_filtering", {}),
                "GeneBalancing": yaml_config.get("gene_preprocessing", {}).get("gene_balancing", {}),
                "GeneShuffle": yaml_config.get("gene_preprocessing", {}).get("gene_shuffle", {})
            },
            "DataProcessing": {
                "Preprocessing": yaml_config.get("data_processing", {}).get("preprocessing", {}),
                "Normalization": yaml_config.get("data_processing", {}).get("normalization", {})
            },
            "DataSplit": {
                "random_state": yaml_config.get("general", {}).get("random_state", 42),
                "test_split": yaml_config.get("data", {}).get("train_test_split", {}).get("test_split", 0.2)
            },
            "ModelParameters": yaml_config.get("model", {}),
            "FeatureImportanceAndVisualizations": yaml_config.get("feature_importance", {}),
            "FileLocations": yaml_config.get("file_locations", {})
        }
        
        return Config(legacy_config)
    
    def validate_config(self) -> None:
        """Validate configuration for required sections and parameters."""
        for section in REQUIRED_CONFIG_SECTIONS:
            if not hasattr(self._config, section):
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Additional validation
        self._validate_model_type()
        self._validate_encoding_variable()
        logger.info("Configuration validation passed")
    
    def _validate_model_type(self) -> None:
        """Validate model type."""
        from ..utils.constants import SUPPORTED_MODEL_TYPES
        model_type = self._config.DataParameters.GeneralSettings.model_type.lower()
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ConfigurationError(f"Invalid model type: {model_type}")
    
    def _validate_encoding_variable(self) -> None:
        """Validate encoding variable."""
        from ..utils.constants import SUPPORTED_ENCODING_VARS
        encoding_var = self._config.DataParameters.GeneralSettings.encoding_variable.lower()
        if encoding_var not in SUPPORTED_ENCODING_VARS:
            raise ConfigurationError(f"Invalid encoding variable: {encoding_var}")
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file."""
        config_dict = self._config.to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {output_path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        # This would recursively update nested dictionaries
        # Implementation depends on specific needs
        pass
    
    @property
    def config(self) -> Config:
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
        # Try to find default config file
        if config_path is None:
            default_paths = [
                "configs/default.yaml",
                "../configs/default.yaml",
                "../../configs/default.yaml"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        _config_manager = ConfigManager(config_path)
        _config_manager.validate_config()
    
    return _config_manager.config


def reset_config() -> None:
    """Reset global configuration instance."""
    global _config_manager
    _config_manager = None