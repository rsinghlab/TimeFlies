"""
Core configuration and management utilities
"""

from .active_config import get_active_project, get_config_for_active_project

__all__ = ["get_active_project", "get_config_for_active_project"]
