# TimeFlies Test Suite

Comprehensive test suite for the TimeFlies project, including unit tests, integration tests, and end-to-end testing.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_runner.py           # Custom test runner script
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_config_manager.py
│   ├── test_path_manager.py
│   └── ...
├── integration/             # Integration tests (slower)
│   ├── test_cli_integration.py
│   └── ...
├── fixtures/                # Test data and utilities
│   ├── sample_data.py       # Sample data generation
│   └── ...
└── README.md               # This file
```

## Quick Start

### Install Test Dependencies
```bash
# Install required packages
pip install pytest pytest-cov pytest-mock pytest-timeout

# Or use the test runner
python tests/test_runner.py --install-deps
```

### Run All Tests
```bash
# Using pytest directly
pytest

# Using test runner
python tests/test_runner.py all
```

### Run Specific Test Types
```bash
# Unit tests only (fast)
python tests/test_runner.py unit

# Integration tests only
python tests/test_runner.py integration

# Fast tests only (excludes slow tests)
python tests/test_runner.py --fast
```

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests for individual components:

- **Configuration Management** (`test_config_manager.py`)
  - Config class functionality
  - YAML loading and validation
  - Global config management

- **Path Management** (`test_path_manager.py`)
  - Directory path construction
  - Naming convention compliance
  - Path validation

- **Model Components** (planned)
  - Model factory functionality
  - Individual model classes
  - Model validation

### Integration Tests (`tests/integration/`)
Tests for component interaction and workflows:

- **CLI Integration** (`test_cli_integration.py`)
  - Command line argument parsing
  - End-to-end command execution
  - Configuration integration

- **Pipeline Integration** (planned)
  - Full pipeline execution
  - Data flow validation
  - Error handling

### Test Markers

Tests are marked for easy selection:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only CLI tests
pytest -m cli

# Skip GPU-dependent tests
pytest -m "not gpu"
```

Available markers:
- `unit`: Fast, isolated unit tests
- `integration`: Integration tests
- `slow`: Tests that take significant time
- `gpu`: Tests requiring GPU
- `data`: Tests requiring real data files
- `cli`: Command line interface tests

## Test Data and Fixtures

### Fixtures (`conftest.py`)
Common test fixtures available in all tests:

- `minimal_config`: Basic valid configuration
- `sample_anndata`: Sample single-cell data
- `temp_project`: Complete temporary project structure
- `temp_dir`: Temporary directory
- `config_file`: Temporary config file

### Sample Data (`fixtures/sample_data.py`)
Utilities for creating test data:

```python
from tests.fixtures.sample_data import create_sample_anndata, TestDataManager

# Create sample single-cell data
adata = create_sample_anndata(n_obs=1000, n_vars=2000)

# Use context manager for full project setup
with TestDataManager() as project_files:
    # project_files contains paths to all created files
    config_path = project_files["config"]
    data_path = project_files["head_train"]
```

## Running Tests

### Command Line Options

#### Using pytest directly:
```bash
# Basic run
pytest

# Verbose output
pytest -v

# Run specific file
pytest tests/unit/test_config_manager.py

# Run specific test
pytest tests/unit/test_config_manager.py::TestConfig::test_config_initialization

# Generate coverage report
pytest --cov=src --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

#### Using test runner:
```bash
# Basic usage
python tests/test_runner.py [unit|integration|all]

# With options
python tests/test_runner.py unit --verbose --coverage

# Fast tests only
python tests/test_runner.py --fast

# Specific marker
python tests/test_runner.py all -m "not gpu"
```

### Continuous Integration

For CI/CD pipelines:

```bash
# Install dependencies
pip install pytest pytest-cov pytest-timeout

# Run tests with coverage
pytest --cov=src --cov-report=xml --cov-report=term-missing

# Run only fast tests in CI
pytest -m "not slow and not data"
```

## Writing Tests

### Test Structure

Follow this structure for new tests:

```python
"""Test description."""

import pytest
from src.timeflies.component import ComponentToTest


class TestComponent:
    """Test ComponentToTest functionality."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        component = ComponentToTest()
        result = component.method()
        assert result == expected_value
        
    def test_error_handling(self):
        """Test error handling."""
        component = ComponentToTest()
        with pytest.raises(ExpectedError):
            component.method_that_should_fail()
```

### Best Practices

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Test Isolation**: Each test should be independent and not rely on other tests
3. **Use Fixtures**: Leverage shared fixtures for common setup
4. **Mark Tests**: Add appropriate markers for test categorization
5. **Mock External Dependencies**: Use mocks for external services or slow operations

### Adding New Tests

1. **Unit Tests**: Add to `tests/unit/test_[component].py`
2. **Integration Tests**: Add to `tests/integration/test_[workflow].py`
3. **Test Data**: Add fixtures to `tests/fixtures/` if needed
4. **Documentation**: Update this README if adding new test categories

## Test Configuration

### pytest.ini
Main pytest configuration:
- Test discovery patterns
- Default options
- Marker definitions
- Timeout settings

### conftest.py
Shared fixtures and pytest hooks:
- Global fixtures
- Environment setup
- Test result reporting
- Marker assignment

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use the test runner which handles this automatically
python tests/test_runner.py
```

#### GPU Tests Failing
```bash
# Skip GPU tests if no GPU available
pytest -m "not gpu"

# Or set environment variable
export CUDA_VISIBLE_DEVICES=""
```

#### Slow Tests
```bash
# Skip slow tests for quick feedback
pytest -m "not slow"

# Or use fast option
python tests/test_runner.py --fast
```

#### Data Dependencies
```bash
# Skip tests requiring real data
pytest -m "not data"

# Or provide test data (see fixtures/sample_data.py)
```

### Debug Mode

For debugging test failures:

```bash
# Drop into debugger on failure
pytest --pdb

# More verbose output
pytest -vv

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

## Coverage Reporting

Generate coverage reports:

```bash
# HTML report (opens in browser)
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=src --cov-report=term-missing

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

## Performance Testing

For performance-sensitive components:

```bash
# Time test execution
pytest --durations=10

# Profile memory usage (if pytest-memray installed)
pytest --memray

# Benchmark tests (if pytest-benchmark installed)
pytest --benchmark-only
```

## Future Enhancements

Planned test improvements:
- Performance benchmarks
- Property-based testing with hypothesis
- Mutation testing
- Visual regression tests for plots
- Load testing for large datasets