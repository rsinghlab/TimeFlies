# TimeFlies Test Suite

Simple, practical test suite for the TimeFlies project. No complexity, just what you need.

## Test Structure

```
tests/
├── conftest.py              # Shared test fixtures
├── test_runner.py           # Simple test runner
├── unit/                    # Fast unit tests
│   ├── test_config_manager.py
│   ├── test_path_manager.py
│   ├── test_model.py
│   └── ...
├── integration/             # Integration tests with real data
│   ├── test_real_data_pipeline.py
│   ├── test_cli_integration.py
│   └── ...
└── fixtures/                # Test data
    ├── test_data.h5ad       # Sample single-cell data
    └── sample_data.py       # Data generation utilities
```

## Running Tests

### Basic Commands
```bash
# Run all tests
python tests/test_runner.py

# Run only unit tests (fast)
python tests/test_runner.py unit

# Run only integration tests
python tests/test_runner.py integration
```

### Practical Options
```bash
# Skip slow tests (quick feedback)
python tests/test_runner.py --fast

# Generate coverage report
python tests/test_runner.py --coverage

# Stop on first failure with details  
python tests/test_runner.py --debug

# Re-run only failed tests
python tests/test_runner.py --rerun

# Show detailed output
python tests/test_runner.py --verbose
```

### Common Combinations
```bash
# Quick unit tests with coverage
python tests/test_runner.py unit --fast --coverage

# Debug failed tests
python tests/test_runner.py --rerun --debug
```

## What Gets Tested

### Unit Tests (`tests/unit/`)
Fast tests for individual components:
- Configuration management
- Path handling
- Model components
- Data processing

### Integration Tests (`tests/integration/`)
Real workflow tests with actual data:
- Complete pipeline execution
- CLI command testing  
- End-to-end data flow

### Test Markers
Simple categories for filtering:
- `unit`: Fast unit tests
- `integration`: Integration tests with real data
- `slow`: Tests that take a while to run

## Test Data

Tests use real `.h5ad` files in the `fixtures/` directory. The integration tests work with actual single-cell data to catch real-world issues that mocked tests might miss.

## Direct pytest Usage

If you prefer using pytest directly:

```bash
# Run all tests
pytest

# Run specific file  
pytest tests/unit/test_model.py

# Generate coverage report
pytest --cov=src --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

## Writing New Tests

### Add Unit Tests
```bash
# Create new file: tests/unit/test_your_component.py
```

### Add Integration Tests  
```bash
# Create new file: tests/integration/test_your_workflow.py
```

### Test Structure Example
```python
import pytest
from src.timeflies.your_component import YourComponent

def test_your_functionality():
    """Test that your component works."""
    component = YourComponent()
    result = component.do_something()
    assert result == expected_value
```

## Troubleshooting

### Tests fail to import modules?
```bash
# Use the test runner (handles Python path automatically)
python tests/test_runner.py
```

### Tests too slow?
```bash
# Skip slow tests
python tests/test_runner.py --fast
```

### Need to debug a failing test?
```bash
# Debug mode with detailed output
python tests/test_runner.py --debug
```