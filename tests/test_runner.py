#!/usr/bin/env python3
"""Test runner script for TimeFlies project."""

import sys
import argparse
import subprocess
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, marker=None):
    """
    Run tests with specified options.
    
    Args:
        test_type: Type of tests to run (unit, integration, all)
        verbose: Enable verbose output
        coverage: Enable coverage reporting  
        marker: Run tests with specific marker
    """
    cmd = ["python", "-m", "pytest"]
    
    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}/")
    
    # Add options
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    if marker:
        cmd.extend(["-m", marker])
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run TimeFlies tests")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "all"],
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage", 
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "-m", "--marker",
        type=str,
        help="Run tests with specific marker (e.g., 'not slow')"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true", 
        help="Run only fast tests (excludes slow and integration tests)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "pytest-cov", "pytest-mock", "pytest-timeout"
        ])
    
    # Set marker for fast tests
    if args.fast:
        args.marker = "not slow and not integration"
    
    # Run tests
    exit_code = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        marker=args.marker
    )
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())