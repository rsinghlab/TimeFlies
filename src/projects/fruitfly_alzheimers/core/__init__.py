"""
Core components for the Fruit Fly Alzheimer's project.

This module contains project-specific configuration and pipeline management.
"""

from .config_manager import Config, ConfigManager, get_config, reset_config
from .pipeline_manager import PipelineManager

__all__ = ['Config', 'ConfigManager', 'get_config', 'reset_config', 'PipelineManager']