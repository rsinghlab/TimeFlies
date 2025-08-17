"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import os
import shutil
import sys
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.fixtures.sample_data import (
    create_sample_anndata, 
    create_test_project_structure,
    create_minimal_config,
    TestDataManager
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory for entire session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def minimal_config():
    """Provide minimal valid configuration."""
    return create_minimal_config()


@pytest.fixture(scope="function") 
def sample_anndata():
    """Create sample AnnData object."""
    return create_sample_anndata(n_obs=100, n_vars=200)


@pytest.fixture(scope="function")
def temp_project():
    """Create temporary project with full structure."""
    with TestDataManager(create_full_structure=True) as project_files:
        yield project_files


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory."""
    with TestDataManager(create_full_structure=False) as project_files:
        yield project_files["base_dir"]


@pytest.fixture(scope="function")
def config_file(temp_dir, minimal_config):
    """Create temporary config file."""
    import yaml
    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(minimal_config, f)
    return str(config_path)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset config manager global state
    try:
        from src.timeflies.core.config_manager import reset_config
        reset_config()
    except ImportError:
        pass
    
    yield
    
    # Clean up after test
    try:
        from src.timeflies.core.config_manager import reset_config
        reset_config()
    except ImportError:
        pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "data: mark test as requiring real data files"
    )
    config.addinivalue_line(
        "markers", "cli: mark test as testing CLI functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        # Add slow marker for certain tests
        if "slow" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            
        # Add CLI marker for CLI tests
        if "cli" in item.name or "cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Disable GPU for testing to avoid conflicts
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
    
    # Set test-specific environment variables
    os.environ["TIMEFLIES_TEST_MODE"] = "1"
    
    yield
    
    # Clean up environment variables
    if "TIMEFLIES_TEST_MODE" in os.environ:
        del os.environ["TIMEFLIES_TEST_MODE"]


# Skip tests requiring GPU if no GPU available
def pytest_runtest_setup(item):
    """Skip GPU tests if no GPU available.""" 
    if "gpu" in item.keywords:
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("TensorFlow not available")


# Custom test result reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom terminal summary."""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        
        terminalreporter.write_sep("=", "TimeFlies Test Summary")
        terminalreporter.write_line(f"Tests passed: {passed}")
        terminalreporter.write_line(f"Tests failed: {failed}")
        terminalreporter.write_line(f"Tests skipped: {skipped}")
        
        if failed == 0:
            terminalreporter.write_line("🎉 All tests passed!", green=True)
        else:
            terminalreporter.write_line(f"❌ {failed} tests failed", red=True)