#!/usr/bin/env python3
"""Test runner for TimeFlies."""

import subprocess
import sys
from pathlib import Path


def run_tests(
    test_type="all",
    verbose=False,
    coverage=False,
    fast=False,
    debug=False,
    rerun_failures=False,
):
    """Run pytest with the given options."""
    cmd = [sys.executable, "-m", "pytest"]

    # Test directory
    if test_type == "all":
        cmd.append("tests/")
    else:
        test_path = Path(f"tests/{test_type}/")
        if test_path.exists():
            cmd.append(str(test_path))
        else:
            print(f"Test directory not found: {test_path}")
            return 1

    # Options
    if debug:
        cmd.extend(["-x", "-v", "--tb=long"])
    elif verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if fast:
        cmd.extend(["-m", "not functional and not system"])

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html:coverage/html", "--cov-report=term"])

    if rerun_failures:
        cmd.extend(["--lf", "-v"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TimeFlies tests")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "functional", "system", "all"],
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--coverage", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--debug", action="store_true", help="Stop on first failure")
    parser.add_argument("--rerun", action="store_true", help="Re-run failed tests only")

    args = parser.parse_args()
    sys.exit(
        run_tests(
            test_type=args.test_type,
            verbose=args.verbose,
            coverage=args.coverage,
            fast=args.fast,
            debug=args.debug,
            rerun_failures=args.rerun,
        )
    )
