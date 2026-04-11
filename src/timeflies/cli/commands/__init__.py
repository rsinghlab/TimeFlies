"""
TimeFlies CLI Commands Package

Each command is implemented in a domain-specific module.
"""

from ._utils import suppress_stderr
from .advanced import queue_command, tune_command
from .analysis import analyze_command, eda_command
from .setup import new_setup_command, split_command
from .testing import create_test_data_command, run_system_tests, run_test_suite
from .training import batch_command, evaluate_command, train_command

__all__ = [
    "execute_command",
    "train_command",
    "evaluate_command",
    "batch_command",
    "eda_command",
    "analyze_command",
    "new_setup_command",
    "split_command",
    "run_system_tests",
    "create_test_data_command",
    "tune_command",
    "queue_command",
    "suppress_stderr",
]


def execute_command(args) -> bool:
    """Execute the appropriate command based on parsed arguments."""
    from ...core.active_config import get_config_for_active_project

    try:
        if args.command == "train":
            config = get_config_for_active_project()
            return train_command(args, config) == 0
        elif args.command == "evaluate":
            config = get_config_for_active_project()
            return evaluate_command(args, config) == 0
        elif args.command == "eda":
            config = get_config_for_active_project()
            return eda_command(args, config) == 0
        elif args.command == "analyze":
            config = get_config_for_active_project()
            return analyze_command(args, config) == 0
        elif args.command == "batch-correct":
            return batch_command(args) == 0
        elif args.command == "setup":
            return new_setup_command(args) == 0
        elif args.command == "split":
            return split_command(args) == 0
        elif args.command == "test":
            return run_system_tests(args) == 0
        elif args.command == "create-test-data":
            return create_test_data_command(args) == 0
        elif args.command == "tune":
            return tune_command(args) == 0
        elif args.command == "queue":
            return queue_command(args) == 0
        else:
            print(f"Unknown command: {args.command}")
            return False
    except Exception as e:
        print(f"Command execution failed: {e}")
        return False
