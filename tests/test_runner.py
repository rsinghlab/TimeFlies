#!/usr/bin/env python3
"""Test runner script for TimeFlies project."""

import sys
import argparse
import subprocess
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, fast=False, debug=False, rerun_failures=False):
    """
    Run tests with practical options only.
    
    Args:
        test_type: Type of tests to run (unit, integration, functional, system, all)
        verbose: Show detailed output
        coverage: Generate HTML coverage report
        fast: Skip slow tests  
        debug: Stop on first failure with detailed output
        rerun_failures: Only re-run tests that failed last time
    """
    # Set environment variables to suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    cmd = ["python3", "-m", "pytest"]
    
    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "functional":
        cmd.append("tests/functional/")
    elif test_type == "system":
        cmd.append("tests/system/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}/")
    
    # Add practical options only
    if fast:
        # Fast mode: only unit and integration tests (skip slow functional tests)
        if test_type == "all":
            cmd = ["python3", "-m", "pytest", "tests/unit/", "tests/integration/"]
        
    if debug:
        cmd.extend(["-x", "-v", "--tb=long"])
    elif verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend([
            "--cov=src/projects",
            "--cov=src/shared", 
            "--cov-report=html:coverage/html",
            "--cov-report=term"
        ])
    
    if rerun_failures:
        cmd.extend(["--lf", "-v"])
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Simplified test runner with only useful options."""
    parser = argparse.ArgumentParser(
        description="Run TimeFlies tests",
        epilog="""
Examples:
  python test_runner.py                    # Run all tests
  python test_runner.py unit               # Unit tests only  
  python test_runner.py --fast             # Skip slow tests
  python test_runner.py --coverage         # With coverage report
  python test_runner.py --debug            # Stop on first failure
  python test_runner.py --rerun            # Re-run failed tests only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "functional", "system", "all"],
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed test output"
    )
    
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true", 
        help="Skip slow tests (quick feedback)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Stop on first failure with full details"
    )
    
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Only re-run tests that failed last time"
    )
    
    args = parser.parse_args()
    
    # Run tests with simplified options
    exit_code = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        fast=args.fast,
        debug=args.debug,
        rerun_failures=args.rerun
    )
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())