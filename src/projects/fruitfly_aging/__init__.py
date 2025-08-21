"""
Fruit Fly Aging Research Project

This module contains aging-specific analysis tools, visualizations,
and specialized methods for studying aging in Drosophila melanogaster.
"""

# Import submodules to ensure they are available as attributes
from . import analysis
from . import evaluation
from . import preprocessing

# Analysis tools
from .analysis import AgingAnalyzer, AgingVisualizer, EDAHandler, VisualizationTools

# Evaluation and metrics
from .evaluation import Interpreter, AgingMetrics

# Preprocessing tools
from .preprocessing import (
    DataSetupManager,
    AgingDataSetupManager,
    DataPreprocessor,
    GeneFilter,
    BatchCorrector,
)

__all__ = [
    # Modules
    "analysis",
    "evaluation",
    "preprocessing",
    # Analysis
    "AgingAnalyzer",
    "AgingVisualizer",
    "EDAHandler",
    "VisualizationTools",
    # Evaluation
    "Interpreter",
    "AgingMetrics",
    # Preprocessing
    "DataSetupManager",
    "AgingDataSetupManager",
    "DataPreprocessor",
    "GeneFilter",
    "BatchCorrector",
]
