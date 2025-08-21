#!/usr/bin/env python3
"""
Development helper script for TimeFlies
Quick access to common development commands
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command with proper error handling."""
    if description:
        print(f"🔄 {description}")
        
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False


def format_code():
    """Format code with Black and isort."""
    print("🎨 Formatting code...")
    success = True
    success &= run_command("python3 -m black src/", "Running Black formatter")
    success &= run_command("python3 -m isort src/", "Running isort")
    return success


def lint_code():
    """Lint code with ruff."""
    print("🔍 Linting code...")
    return run_command("python3 -m ruff check src/", "Running ruff linter")


def type_check():
    """Type check with mypy."""
    print("🔎 Type checking...")
    return run_command("python3 -m mypy src/", "Running mypy")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    return run_command("python3 run_timeflies.py test --fast", "Running test suite")


def run_all():
    """Run all checks: format, lint, type-check, test."""
    print("🚀 Running all development checks...")
    
    steps = [
        ("Format", format_code),
        ("Lint", lint_code), 
        ("Type Check", type_check),
        ("Test", run_tests)
    ]
    
    results = {}
    for name, func in steps:
        results[name] = func()
        print()
    
    print("📋 Results:")
    all_passed = True
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed!")
        return True
    else:
        print("\n💥 Some checks failed")
        return False


def clean():
    """Clean cache files."""
    print("🧹 Cleaning cache files...")
    cache_dirs = [".pytest_cache", ".mypy_cache", ".ruff_cache", "coverage", "__pycache__"]
    
    for cache_dir in cache_dirs:
        run_command(f"find . -name '{cache_dir}' -type d -exec rm -rf {{}} + 2>/dev/null || true")
    
    run_command("find . -name '*.pyc' -delete")
    print("✅ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="TimeFlies Development Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 dev.py format     # Format code with Black + isort
  python3 dev.py lint       # Lint with ruff
  python3 dev.py test       # Run fast tests
  python3 dev.py all        # Run all checks
  python3 dev.py clean      # Clean cache files
        """
    )
    
    parser.add_argument(
        'command',
        choices=['format', 'lint', 'type', 'test', 'all', 'clean'],
        help='Development command to run'
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Run this script from the TimeFlies root directory")
        sys.exit(1)
    
    commands = {
        'format': format_code,
        'lint': lint_code,
        'type': type_check, 
        'test': run_tests,
        'all': run_all,
        'clean': clean
    }
    
    success = commands[args.command]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()