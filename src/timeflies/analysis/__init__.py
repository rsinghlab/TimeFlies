"""Shared analysis components for TimeFlies projects."""

from .eda import EDAHandler
from .visualizer import DataVisualizer
from .visuals import Visualizer

# Backward compatibility alias
AgingVisualizer = DataVisualizer

__all__ = ["EDAHandler", "DataVisualizer", "AgingVisualizer", "Visualizer"]
