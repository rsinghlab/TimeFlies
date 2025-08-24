"""Fruit Fly Aging project-specific configuration management."""

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from common.utils.exceptions import ConfigurationError
from common.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global config manager instance for singleton pattern
_config_manager: Optional["ConfigManager"] = None


class Config:
    """Configuration class with nested attribute access for aging project."""

    def __init__(self, config_dict: dict[str, Any]):
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

    def to_dict(self) -> dict[str, Any]:
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

    def update(self, other: dict[str, Any]) -> None:
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

    def __init__(
        self, config_path: str | None = None, project_override: str | None = None
    ):
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
            except (ImportError, Exception):
                # Fallback to aging project if detection fails
                self.project_name = "fruitfly_aging"
        # Note: User configs (config.yaml) are created during 'timeflies setup' command

        self.config_path = self._find_config_path(config_path)
        self._config = None
        self._load_config()
        self.validate_config()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """
        Create a Config object directly from a dictionary.

        Args:
            config_dict: Dictionary containing configuration data

        Returns:
            Config object initialized with the provided dictionary
        """
        return Config(config_dict)

    def _find_config_path(self, config_path: str | None) -> str:
        """Find configuration file path for aging project."""
        if config_path:
            if os.path.exists(config_path):
                return config_path
            else:
                raise ConfigurationError(f"Configuration file not found: {config_path}")

        # Priority order: local config.yaml > user ~/.timeflies/configs/ > dev configs
        user_config_dir = Path.home() / ".timeflies" / "configs"
        default_paths = [
            "./config.yaml",  # Project-local config (highest priority)
            str(user_config_dir / "default.yaml"),  # User's installed config
            "configs/default.yaml",  # Dev config (development only)
            "../configs/default.yaml",
            "../../configs/default.yaml",
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
            with open(self.config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            # Check if there's a base config to inherit from
            base_config_path = config_dict.get("base_config")
            if base_config_path:
                # Resolve relative path from config file location
                config_dir = os.path.dirname(self.config_path)
                base_config_full_path = os.path.join(config_dir, base_config_path)

                if os.path.exists(base_config_full_path):
                    # Load base config
                    with open(base_config_full_path, encoding="utf-8") as bf:
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
        self, base_dict: dict[str, Any], override_dict: dict[str, Any]
    ) -> dict[str, Any]:
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

    def _ensure_user_configs(self) -> None:
        """Ensure user has a config.yaml in their project directory."""
        import shutil

        # Skip config creation during tests or CI
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
            return

        # Check if user already has config.yaml in current directory
        project_config = Path("./config.yaml")
        if project_config.exists():
            return  # User already has their config

        # Copy default config to user's project directory
        print("📋 Creating config.yaml in your project directory...")
        if not project_config.exists():
            # Find source config from package/development
            source_configs = [
                Path(__file__).parent.parent.parent
                / "configs"
                / "default.yaml",  # dev path
                Path(__file__).parent
                / "configs"
                / "default.yaml",  # potential package path
            ]

            for source_config in source_configs:
                if source_config.exists():
                    shutil.copy2(source_config, project_config)
                    print("✅ Created config.yaml from TimeFlies defaults")
                    print(
                        "📝 You can now edit ./config.yaml to customize your project settings"
                    )
                    return  # Success - exit method

            # If no source config found, create from current repo's config
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "find",
                        "/home/nikolaitennant/projects/TimeFlies",
                        "-name",
                        "default.yaml",
                        "-path",
                        "*/configs/*",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    source_path = result.stdout.strip().split("\n")[0]
                    shutil.copy2(source_path, project_config)
                    print(f"✅ Created config.yaml from {source_path}")
                    print(
                        "📝 You can now edit ./config.yaml to customize your project settings"
                    )
                    return
            except:
                pass
                # Create minimal default config if no source found
                default_config = project_config
                minimal_config = {
                    "project": "fruitfly_alzheimers",
                    "general": {"random_seed": 42, "log_level": "INFO"},
                    "data": {
                        "tissue": "head",
                        "model": "cnn",
                        "target_variable": "age",
                        "batch_correction": {"enabled": False},
                        "sampling": {"samples": 10000, "variables": None},
                        "split": {
                            "method": "column",
                            "column": "genotype",
                            "train": ["control"],
                            "test": ["ab42", "htau"],
                        },
                    },
                    "model": {
                        "cnn": {"filters": 32, "kernel_size": 48},
                        "training": {
                            "epochs": 100,
                            "batch_size": 32,
                            "validation_split": 0.2,
                        },
                    },
                }
                with open(default_config, "w") as f:
                    yaml.dump(minimal_config, f, default_flow_style=False)
                logger.info(f"Created minimal config at {default_config}")

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


def get_config(config_path: str | None = None) -> Config:
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


def get_config_manager(config_path: str | None = None) -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_path)

    return _config_manager


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config_manager
    _config_manager = None
