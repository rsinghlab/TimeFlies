"""
Active project configuration management.

This module handles the active project selection and automatic config loading.
It reads the active_project.yaml file to determine which project config to use.
"""

import os
from typing import Any

import yaml

from ..utils.exceptions import ConfigurationError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def get_active_project() -> str:
    """
    Get the currently active project from active_project.yaml.

    Returns:
        str: The active project name (e.g., 'fruitfly_aging')
    """
    # Try to find active_project.yaml
    search_paths = [
        "configs/active_project.yaml",
        "../configs/active_project.yaml",
        "../../configs/active_project.yaml",
        os.path.join(os.path.dirname(__file__), "../../../configs/active_project.yaml"),
    ]

    active_config_path = None
    for path in search_paths:
        if os.path.exists(path):
            active_config_path = path
            break

    if not active_config_path:
        # Fallback: try to read project from main default.yaml
        try:
            default_paths = [
                "configs/default.yaml",
                "../configs/default.yaml",
                "../../configs/default.yaml",
                os.path.join(
                    os.path.dirname(__file__), "../../../configs/default.yaml"
                ),
            ]

            for path in default_paths:
                if os.path.exists(path):
                    with open(path) as f:
                        default_config = yaml.safe_load(f)
                    project = default_config.get("project")
                    if project:
                        logger.debug(f"Active project from default.yaml: {project}")
                        return project
                    break

            raise ConfigurationError(
                "No active_project.yaml found and no 'project' in default.yaml! "
                "Please create configs/active_project.yaml or set 'project' in configs/default.yaml"
            )
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise ConfigurationError(
                f"Error reading configuration: {e}. "
                "Please create configs/active_project.yaml "
                "and set 'active_project' to either 'fruitfly_aging' or 'fruitfly_alzheimers'"
            )

    try:
        with open(active_config_path) as f:
            active_config = yaml.safe_load(f)

        active_project = active_config.get("active_project")
        if not active_project:
            raise ConfigurationError(
                "No 'active_project' specified in active_project.yaml"
            )

        logger.info(f"Active project: {active_project}")
        return active_project

    except Exception as e:
        raise ConfigurationError(f"Error reading active_project.yaml: {e}")


def get_active_config_path(config_type: str = "default") -> str:
    """
    Get the config path for the active project.

    Args:
        config_type: Type of config ('default', 'setup', 'batch', etc.)

    Returns:
        str: Path to the config file for the active project
    """
    get_active_project()

    # Map config types to file names
    config_files = {
        "default": "default.yaml",
        "setup": "setup.yaml",
        "batch": "batch_config.yaml",
    }

    config_filename = config_files.get(config_type, f"{config_type}.yaml")

    # Use unified config structure (no more project-specific folders)
    config_path = f"configs/{config_filename}"

    # Check if it exists
    if not os.path.exists(config_path):
        # Try relative paths
        search_paths = [
            config_path,
            f"../{config_path}",
            f"../../{config_path}",
            os.path.join(os.path.dirname(__file__), f"../../../{config_path}"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            raise ConfigurationError(
                f"Config file not found: {config_filename} (looked in configs/ directory)"
            )

    logger.info(f"Using config: {config_path}")
    return config_path


def get_quick_overrides() -> dict[str, Any]:
    """
    Get any quick overrides from active_project.yaml.

    Returns:
        dict: Quick override settings
    """
    # Find active_project.yaml
    search_paths = [
        "configs/active_project.yaml",
        "../configs/active_project.yaml",
        "../../configs/active_project.yaml",
        os.path.join(os.path.dirname(__file__), "../../../configs/active_project.yaml"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            with open(path) as f:
                active_config = yaml.safe_load(f)
                return active_config.get("quick_overrides", {})

    return {}


def get_config_for_active_project(config_type: str = "default"):
    """
    Get the configuration for the active project.

    Args:
        config_type: Type of config to load ('default', 'setup', etc.)

    Returns:
        Config instance for the active project with overrides applied
    """
    get_active_project()
    config_path = get_active_config_path(config_type)

    # Use shared config manager for all projects
    from .config_manager import ConfigManager

    # Load the config
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    # Apply any quick overrides
    overrides = get_quick_overrides()
    if overrides:
        logger.info(f"Applying quick overrides: {overrides}")
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "model.epochs"
                parts = key.split(".")
                current = config
                for part in parts[:-1]:
                    current = getattr(current, part)
                setattr(current, parts[-1], value)
            else:
                # Simple top-level override
                if hasattr(config, "data") and hasattr(config.data, key):
                    setattr(config.data, key, value)
                elif hasattr(config, "model") and hasattr(config.model, key):
                    setattr(config.model, key, value)
                else:
                    setattr(config, key, value)

    return config
