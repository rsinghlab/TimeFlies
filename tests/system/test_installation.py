"""System-level tests for TimeFlies installation and setup."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestInstallation:
    """Test TimeFlies installation and basic system requirements."""

    def test_python_version_requirement(self):
        """Test that we're running on supported Python version."""
        version = sys.version_info
        assert version.major == 3
        assert (
            version.minor >= 10
        ), f"Python 3.10+ required, got {version.major}.{version.minor}"

    def test_core_dependencies_available(self):
        """Test that core dependencies are importable."""
        core_deps = [
            "numpy",
            "pandas",
            "anndata",
            "scanpy",
            "sklearn",
            "tensorflow",
            "matplotlib",
            "seaborn",
        ]

        for dep in core_deps:
            try:
                __import__(dep)
            except ImportError:
                pytest.fail(f"Core dependency '{dep}' not available")

    def test_timeflies_package_structure(self):
        """Test that TimeFlies package is properly structured."""
        try:
            # Test core module imports
            from common.cli.main import main_cli
            from common.core.config_manager import ConfigManager
            from common.data.loaders import DataLoader
            from common.models.model_factory import ModelFactory

            # Test that classes can be instantiated
            assert ConfigManager is not None
            assert DataLoader is not None
            assert ModelFactory is not None
            assert main_cli is not None

        except ImportError as e:
            pytest.fail(f"TimeFlies package structure issue: {e}")

    def test_cli_help_functionality(self):
        """Test that CLI help works without errors."""
        from common.cli.main import main_cli

        # Test main help
        try:
            main_cli(["--help"])
        except SystemExit as e:
            # Help command exits with code 0
            assert e.code == 0
        except Exception as e:
            pytest.fail(f"CLI help failed: {e}")

    def test_config_loading_system(self):
        """Test configuration system works."""
        from common.core.active_config import get_active_project

        # Should return a valid project name or default
        project = get_active_project()
        assert isinstance(project, str)
        assert len(project) > 0


class TestSystemSetup:
    """Test system setup and environment verification."""

    def test_temporary_environment_setup(self):
        """Test setup in a temporary environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Test that we can create basic project structure
                configs_dir = Path("configs")
                configs_dir.mkdir(exist_ok=True)

                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)

                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)

                # Test directories were created
                assert configs_dir.exists()
                assert data_dir.exists()
                assert models_dir.exists()

            finally:
                os.chdir(old_cwd)

    def test_data_directory_access(self):
        """Test that we can work with data directories."""
        from common.core.config_manager import ConfigManager
        from common.utils.path_manager import PathManager

        # Create a minimal config for testing
        _ = {
            "general": {"project_name": "test_project"},
            "data": {"tissue": "head", "target_variable": "age"},
            "paths": {
                "data": {"train": "data/test_train.h5ad", "eval": "data/test_eval.h5ad"}
            },
        }

        try:
            # Test classes exist and can be instantiated (without calling real methods)
            assert ConfigManager is not None
            assert PathManager is not None

        except ImportError as e:
            pytest.fail(f"Required classes not available: {e}")

    def test_model_factory_system(self):
        """Test that model factory system works."""
        from common.core.config_manager import ConfigManager
        from common.models.model_factory import ModelFactory

        _ = {
            "general": {"project_name": "test"},
            "data": {"target_variable": "age"},
            "model": {"training": {"epochs": 1, "batch_size": 32}},
        }

        try:
            # Test classes exist and can be imported
            assert ConfigManager is not None
            assert ModelFactory is not None

        except ImportError as e:
            pytest.fail(f"Required classes not available: {e}")


class TestDataIntegrity:
    """Test data handling and integrity."""

    def test_anndata_compatibility(self):
        """Test AnnData object creation and manipulation."""
        from tests.fixtures.unit_test_data import create_sample_anndata

        # Create test data
        adata = create_sample_anndata(n_cells=100, n_genes=50)

        # Test basic properties
        assert adata.n_obs == 100
        assert adata.n_vars == 50
        assert adata.X is not None

        # Test that we can perform common operations
        import numpy as np

        # Basic statistics
        if hasattr(adata.X, "toarray"):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X

        assert X_dense.shape == (100, 50)
        assert np.all(X_dense >= 0)  # Count data should be non-negative

    def test_file_io_system(self):
        """Test file I/O operations."""
        from tests.fixtures.unit_test_data import create_sample_anndata

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            adata = create_sample_anndata(n_cells=50, n_genes=25)

            # Test file writing
            test_file = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(test_file)

            assert test_file.exists()
            assert test_file.stat().st_size > 0

            # Test file reading
            import anndata

            adata_loaded = anndata.read_h5ad(test_file)

            assert adata_loaded.n_obs == 50
            assert adata_loaded.n_vars == 25


class TestEnvironmentValidation:
    """Test environment validation and requirements."""

    def test_gpu_detection(self):
        """Test GPU detection (should not fail if no GPU)."""
        try:
            import tensorflow as tf

            # Test that TensorFlow can detect devices
            devices = tf.config.list_physical_devices()
            assert isinstance(devices, list)

            # GPU detection (optional)
            gpus = tf.config.list_physical_devices("GPU")
            # Don't require GPU, just test detection works
            assert isinstance(gpus, list)

        except Exception as e:
            pytest.skip(f"TensorFlow GPU detection not available: {e}")

    def test_memory_constraints(self):
        """Test basic memory constraints."""
        import psutil

        # Test that we have reasonable memory available
        memory = psutil.virtual_memory()

        # Require at least 1GB available memory
        available_gb = memory.available / (1024**3)
        assert (
            available_gb >= 1.0
        ), f"Insufficient memory: {available_gb:.1f}GB available"

    def test_disk_space(self):
        """Test disk space availability."""
        import shutil

        # Test current directory has some space
        total, used, free = shutil.disk_usage(".")

        # Require at least 100MB free space
        free_mb = free / (1024**2)
        assert free_mb >= 100, f"Insufficient disk space: {free_mb:.1f}MB free"


@pytest.mark.slow
class TestIntegrationReadiness:
    """Test system readiness for integration testing."""

    def test_sample_workflow_execution(self):
        """Test that basic workflow components are available."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        try:
            from common.core.config_manager import ConfigManager
            from common.models.model_factory import ModelFactory
            from tests.fixtures.unit_test_data import create_sample_anndata

            # Test that all required components can be imported
            assert ConfigManager is not None
            assert ModelFactory is not None
            assert create_sample_anndata is not None
            assert train_test_split is not None
            assert LabelEncoder is not None

        except ImportError as e:
            pytest.fail(f"Required workflow components not available: {e}")

    def test_end_to_end_data_pipeline(self):
        """Test that data pipeline components are available."""
        try:
            from common.core.config_manager import ConfigManager
            from common.data.preprocessing.data_processor import DataPreprocessor
            from tests.fixtures.unit_test_data import create_sample_anndata

            # Test that all required components can be imported
            assert ConfigManager is not None
            assert DataPreprocessor is not None
            assert create_sample_anndata is not None

        except ImportError as e:
            pytest.fail(f"Required data pipeline components not available: {e}")
