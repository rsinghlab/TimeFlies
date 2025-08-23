#!/usr/bin/env python3
"""Test runner script for TimeFlies project."""

import sys
import argparse
import subprocess
from pathlib import Path
import os


def get_batch_correction_tests():
    """Find all batch correction related test files."""
    batch_tests = []
    
    # Look for batch correction specific tests
    test_dirs = ['tests/unit/', 'tests/integration/']
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            # Find test files that mention batch correction
            for test_file in test_path.glob('test_*batch*.py'):
                batch_tests.append(str(test_file))
            # Also check for batch correction in file content
            for test_file in test_path.glob('test_*.py'):
                try:
                    content = test_file.read_text()
                    if 'batch_correction' in content.lower() or 'batchcorrector' in content:
                        batch_tests.append(str(test_file))
                except:
                    continue
    
    return list(set(batch_tests))  # Remove duplicates


def has_batch_environment():
    """Check if .venv_batch environment exists."""
    return Path('.venv_batch').exists()


def run_in_batch_environment(test_files, cmd_options):
    """Run specific tests in batch correction environment."""
    if not test_files:
        return 0
        
    if not has_batch_environment():
        print("⚠️  Batch correction environment (.venv_batch) not found")
        print("   Skipping batch correction tests")
        print("   Run 'python3 run_timeflies.py setup --dev' to create it")
        return 0
    
    print(f"\n🔄 Switching to batch correction environment for {len(test_files)} test(s)...")
    
    # Build command to run in batch environment
    batch_python = ".venv_batch/bin/python" if Path(".venv_batch/bin/python").exists() else ".venv_batch/Scripts/python.exe"
    
    cmd = [batch_python, "-m", "pytest"] + test_files + cmd_options
    
    print(f"   Running: {' '.join(cmd)}")
    
    # Temporarily set batch environment variables
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = str(Path('.venv_batch').resolve())
    env['PATH'] = f"{Path('.venv_batch/bin').resolve()}:{env.get('PATH', '')}"
    
    result = subprocess.run(cmd, env=env)
    
    print("🔄 Returning to main environment...")
    
    return result.returncode


def run_tests(test_type="all", verbose=False, coverage=False, fast=False, debug=False, rerun_failures=False):
    """
    Run tests with automatic environment switching for batch correction tests.
    
    Args:
        test_type: Type of tests to run (unit, integration, functional, system, all)
        verbose: Show detailed output
        coverage: Generate HTML coverage report
        fast: Skip slow tests  
        debug: Stop on first failure with detailed output
        rerun_failures: Only re-run tests that failed last time
    """
    # Set environment variables to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Get all test files to run
    all_test_files = []
    if test_type == "unit":
        all_test_files = list(Path("tests/unit/").glob("test_*.py"))
    elif test_type == "integration": 
        all_test_files = list(Path("tests/integration/").glob("test_*.py"))
    elif test_type == "functional":
        all_test_files = list(Path("tests/functional/").glob("test_*.py"))
    elif test_type == "system":
        all_test_files = list(Path("tests/system/").glob("test_*.py"))
    elif test_type == "all":
        for test_dir in ["unit", "integration", "functional", "system"]:
            test_path = Path(f"tests/{test_dir}/")
            if test_path.exists():
                all_test_files.extend(test_path.glob("test_*.py"))
    else:
        test_path = Path(f"tests/{test_type}/")
        if test_path.exists():
            all_test_files.extend(test_path.glob("test_*.py"))
    
    # Fast mode: skip functional tests
    if fast and test_type == "all":
        all_test_files = [f for f in all_test_files if "functional" not in str(f)]
    
    # Separate batch correction tests from regular tests
    batch_tests = get_batch_correction_tests()
    batch_test_files = [f for f in all_test_files if str(f) in batch_tests]
    regular_test_files = [f for f in all_test_files if str(f) not in batch_tests]
    
    # Build pytest options
    pytest_options = []
    if debug:
        pytest_options.extend(["-x", "-v", "--tb=long"])
    elif verbose:
        pytest_options.append("-v")
    else:
        pytest_options.append("-q")
    
    if coverage:
        pytest_options.extend([
            "--cov=src",
            "--cov-report=html:coverage/html",
            "--cov-report=term",
            "--cov-append"  # Append coverage from multiple runs
        ])
    
    if rerun_failures:
        pytest_options.extend(["--lf", "-v"])
    
    # Track overall success
    overall_success = True
    
    # 1. Run regular tests in main environment
    if regular_test_files:
        print(f"🧪 Running {len(regular_test_files)} regular tests in main environment...")
        
        cmd = ["python3", "-m", "pytest"] + [str(f) for f in regular_test_files] + pytest_options
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            overall_success = False
    
    # 2. Run batch correction tests in batch environment
    if batch_test_files:
        batch_result = run_in_batch_environment(
            [str(f) for f in batch_test_files], 
            pytest_options
        )
        if batch_result != 0:
            overall_success = False
    
    # 3. Show summary
    if batch_test_files and regular_test_files:
        print(f"\n📋 Test Summary:")
        print(f"   Regular tests: {len(regular_test_files)} files")
        print(f"   Batch correction tests: {len(batch_test_files)} files")
        print(f"   Status: {'✅ All passed' if overall_success else '❌ Some failed'}")
    
    return 0 if overall_success else 1


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