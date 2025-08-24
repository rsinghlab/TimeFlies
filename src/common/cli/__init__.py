"""TimeFlies Command Line Interface module."""

from .commands import (
    analyze_command,
    batch_command,
    eda_command,
    evaluate_command,
    execute_command,
    new_setup_command,
    run_system_tests,
    split_command,
    train_command,
)
from .main import main_cli

__all__ = [
    "main_cli",
    "execute_command",
    "train_command",
    "evaluate_command",
    "eda_command",
    "analyze_command",
    "batch_command",
    "new_setup_command",
    "split_command",
    "run_system_tests",
]
