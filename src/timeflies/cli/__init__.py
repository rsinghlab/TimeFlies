"""TimeFlies Command Line Interface module."""

from .main import main_cli
from .commands import (
    train_command,
    evaluate_command,
    batch_command,
    setup_command,
    test_command,
    validate_command
)

__all__ = [
    'main_cli',
    'train_command',
    'evaluate_command', 
    'batch_command',
    'setup_command',
    'test_command',
    'validate_command'
]