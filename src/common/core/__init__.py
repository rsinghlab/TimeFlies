"""Shared core components for TimeFlies projects."""

from .active_config import get_active_project, get_config_for_active_project
from .config_manager import Config, ConfigManager
from .pipeline_manager import PipelineManager

__all__ = ['PipelineManager', 'Config', 'ConfigManager', 'get_config_for_active_project', 'get_active_project']
