"""
Fruit Fly Aging Research Project

This module contains aging-specific analysis tools for studying aging in Drosophila melanogaster.
"""

# Import submodules to ensure they are available as attributes
from . import analysis

# Analysis tools
from .analysis import AgingAnalyzer

__all__ = [
    # Modules
    "analysis",
    # Analysis
    "AgingAnalyzer",
]
