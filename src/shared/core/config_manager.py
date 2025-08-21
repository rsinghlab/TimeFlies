"""Fruit Fly Aging project-specific configuration management."""

import os
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from shared.utils.exceptions import ConfigurationError
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global config manager instance for singleton pattern
_config_manager: Optional["ConfigManager"] = None


class Config:
    """Configuration class with nested attribute access for aging project."""

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
        if name.startswith("_"):
            raise AttributeError(f"'{name}' not found")
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"Configuration '{name}' not found")

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting configuration values."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, "_data"):
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
            if (
                isinstance(value, dict)
                and key in self._data
                and isinstance(self._data[key], Config)
            ):
                self._data[key].update(value)
            else:
                if isinstance(value, dict):
                    value = Config(value)
                self._data[key] = value

    def __repr__(self) -> str:
        return f"Config({self._data})"


class ConfigManager:
    """Fruit Fly Aging project-specific configuration manager."""

    def __init__(self, config_path: Optional[str] = None, project_override: Optional[str] = None):
        """
        Initialize configuration manager for aging project.

        Args:
            config_path: Path to YAML configuration file
            project_override: Override the default project name for this session
        """
        if project_override:
            self.project_name = project_override
        else:
            # Auto-detect active project from config system
            try:
                from .active_config import get_active_project
                self.project_name = get_active_project()
            except:
                # Fallback to aging project if detection fails
                self.project_name = "fruitfly_aging"
        self.config_path = self._find_config_path(config_path)
        self._config = None
        self._load_config()
        self.validate_config()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create a Config object directly from a dictionary.

        Args:
            config_dict: Dictionary containing configuration data

        Returns:
            Config object initialized with the provided dictionary
        """
        return Config(config_dict)

    def _find_config_path(self, config_path: Optional[str]) -> str:
        """Find configuration file path for aging project."""
        if config_path:
            if os.path.exists(config_path):
                return config_path
            else:
                raise ConfigurationError(f"Configuration file not found: {config_path}")

        # For project overrides, use the global default.yaml
        # For normal usage, try project-specific configs first
        default_paths = [
            "configs/default.yaml",  # Global default for project overrides
            f"configs/{self.project_name}/default.yaml",
            f"../configs/{self.project_name}/default.yaml", 
            f"../../configs/{self.project_name}/default.yaml",
            "../configs/default.yaml",
            "../../configs/default.yaml",
            os.path.join(
                os.path.dirname(__file__),
                f"../../../configs/{self.project_name}/default.yaml",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "../../../configs/default.yaml",
            ),
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        raise ConfigurationError(
            f"No configuration file found for {self.project_name} project. Please provide a valid config path."
        )

    def _load_config(self) -> None:
        """Load configuration from YAML file with base config support."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            # Check if there's a base config to inherit from
            base_config_path = config_dict.get("base_config")
            if base_config_path:
                # Resolve relative path from config file location
                config_dir = os.path.dirname(self.config_path)
                base_config_full_path = os.path.join(config_dir, base_config_path)

                if os.path.exists(base_config_full_path):
                    # Load base config
                    with open(base_config_full_path, "r", encoding="utf-8") as bf:
                        base_config_dict = yaml.safe_load(bf)

                    # Merge base config with project config (project config overrides base)
                    config_dict = self._merge_configs(base_config_dict, config_dict)
                    logger.info(f"Merged base config from {base_config_full_path}")
                else:
                    logger.warning(
                        f"Base config file not found: {base_config_full_path}"
                    )

                # Remove the base_config reference from final config
                config_dict.pop("base_config", None)

            # Ensure project name is set in both main config and data section
            config_dict["project"] = self.project_name
            if "data" not in config_dict:
                config_dict["data"] = {}
            config_dict["data"]["project"] = self.project_name
            
            # Note: Project-specific splitting configurations are now in project config files

            # Create Config object
            self._config = Config(config_dict)
            logger.info(
                f"Loaded {self.project_name} configuration from {self.config_path}"
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load {self.project_name} config from {self.config_path}: {e}"
            )

    def _merge_configs(
        self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = base_dict.copy()

        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override or add new key
                result[key] = value

        return result

    def validate_config(self) -> None:
        """Validate configuration for required sections and parameters."""
        required_sections = ["data", "general"]

        for section in required_sections:
            if not hasattr(self._config, section):
                raise ConfigurationError(
                    f"Missing required configuration section: {section}"
                )

        # Aging project specific validations
        if hasattr(self._config, "data"):
            # Ensure project is set correctly
            if not hasattr(self._config.data, "project"):
                self._config.data.project = self.project_name
            elif self._config.data.project != self.project_name:
                logger.warning(
                    f"Config project '{self._config.data.project}' doesn't match expected '{self.project_name}'"
                )
                self._config.data.project = self.project_name

    def get_config(self) -> Config:
        """Get the loaded configuration."""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        return self._config

    def save_config(self, config: Config, output_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            logger.info(f"Saved {self.project_name} configuration to {output_path}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save {self.project_name} config to {output_path}: {e}"
            )


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance for aging project (singleton pattern).

    Args:
        config_path: Path to configuration file (only used for first call)

    Returns:
        Config: Global configuration instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_path)

    return _config_manager.get_config()


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_path)

    return _config_manager


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config_manager
    _config_manager = None
