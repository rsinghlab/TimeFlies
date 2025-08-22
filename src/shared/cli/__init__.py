"""TimeFlies Command Line Interface module."""

from .main import main_cli
from .commands import (
    train_command,
    evaluate_command,
    eda_command,
    analyze_command,
    batch_command,
    setup_command,
    run_system_tests,
)

__all__ = [
    "main_cli",
    "train_command",
    "evaluate_command",
    "eda_command",
    "analyze_command",
    "batch_command",
    "setup_command",
    "run_system_tests",
]
