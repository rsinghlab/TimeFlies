#!/usr/bin/env python3
"""CLI entry point for TimeFlies."""

import logging
import os
import warnings

# Suppress TensorFlow/CUDA noise before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["GLOG_minloglevel"] = "3"

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    """Entry point for the timeflies command."""
    try:
        from timeflies.cli.commands import execute_command
        from timeflies.cli.parser import create_main_parser

        parser = create_main_parser()
        args = parser.parse_args()

        success = execute_command(args)
        raise SystemExit(0 if success else 1)

    except ImportError as e:
        print(f"Error importing TimeFlies modules: {e}")
        print("Make sure TimeFlies is properly installed:")
        print("  uv pip install git+https://github.com/rsinghlab/TimeFlies")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
