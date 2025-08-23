"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import os
import shutil
import sys
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.fixtures.unit_test_data import (
    create_sample_anndata, 
    create_test_project_structure,
    create_minimal_config,
    TestDataManager
)

# Import project modules for fixtures
# from projects.fruitfly_aging.core.config_manager import Config, ConfigManager
from shared.cli.parser import create_main_parser
from shared.core.active_config import get_config_for_active_project
from unittest.mock import Mock, patch


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


@pytest.fixture(scope="function")
def config_object(minimal_config):
    """Create Config object from minimal config."""
    return Config(minimal_config)


@pytest.fixture(scope="function")
def config_manager(config_object):
    """Create ConfigManager with test config."""
    return ConfigManager(config_dict=config_object.to_dict())


@pytest.fixture(scope="function")
def cli_parser():
    """Create CLI parser for testing."""
    return create_main_parser()


@pytest.fixture(scope="function")
def mock_args():
    """Create mock args object for CLI testing."""
    args = Mock()
    args.verbose = False
    args.project = None
    args.command = 'test'
    return args


@pytest.fixture(scope="function")
def sample_data_files(temp_dir):
    """Create sample data files for testing."""
    from tests.fixtures.unit_test_data import create_sample_h5ad_files
    return create_sample_h5ad_files(temp_dir)


@pytest.fixture(scope="function")
def large_sample_anndata():
    """Create larger sample AnnData for integration tests."""
    return create_sample_anndata(n_obs=1000, n_vars=2000)


@pytest.fixture(scope="function")
def small_sample_anndata():
    """Create small sample AnnData for unit tests."""
    return create_sample_anndata(n_obs=50, n_vars=100)


@pytest.fixture(scope="function")
def aging_config():
    """Create aging project config."""
    config_dict = {
        'general': {'project_name': 'TimeFlies', 'version': '0.2.0', 'random_state': 42},
        'data': {
            'tissue': 'head',
            'target_variable': 'age',
            'model': 'CNN',
            'species': 'drosophila',
            'cell_type': 'all',
            'sex': 'all',
            'batch_correction': {'enabled': False},
            'filtering': {'include_mixed_sex': False},
            'sampling': {'samples': 1000, 'variables': None},
            'split': {
                'method': 'random',
                'test_ratio': 0.1,
                'sex': {'train': 'male', 'test': 'female', 'test_ratio': 0.3},
                'tissue': {'train': 'head', 'test': 'body', 'test_ratio': 0.3}
            }
        },
        'paths': {
            'data': {
                'train': 'data/{project}/{tissue}/{species}_{tissue}_{project}_train.h5ad',
                'eval': 'data/{project}/{tissue}/{species}_{tissue}_{project}_eval.h5ad',
                'original': 'data/{project}/{tissue}/{species}_{tissue}_{project}_original.h5ad'
            },
            'batch_data': {
                'train': 'data/{project}/{tissue}/{species}_{tissue}_{project}_train_batch.h5ad',
                'eval': 'data/{project}/{tissue}/{species}_{tissue}_{project}_eval_batch.h5ad'
            }
        },
        'model': {
            'training': {'epochs': 10, 'batch_size': 32, 'validation_split': 0.2, 'early_stopping_patience': 8, 'learning_rate': 0.001}
        },
        'preprocessing': {
            'genes': {'remove_sex_genes': False, 'highly_variable_genes': False},
            'balancing': {'balance_genes': False},
            'shuffling': {'shuffle_genes': False}
        },
        'gene_preprocessing': {
            'gene_filtering': {'remove_sex_genes': False, 'highly_variable_genes': False},
            'gene_balancing': {'balance_genes': False},
            'gene_shuffle': {'shuffle_genes': False, 'shuffle_random_state': 42}
        },
        'analysis': {'normalization': {'enabled': False}, 'eda': {'enabled': False}},
        'data_processing': {'normalization': {'enabled': False}, 'exploratory_data_analysis': {'enabled': False}},
        'interpretation': {'shap': {'enabled': False, 'load_existing': False, 'reference_size': 100}},
        'visualizations': {'enabled': True},
        'hardware': {'processor': 'GPU', 'memory_growth': True},
        'logging': {'level': 'INFO', 'format': 'detailed', 'to_file': False}
    }
    return Config(config_dict)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset any global state if needed
    yield
    
    # Clean up after test
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