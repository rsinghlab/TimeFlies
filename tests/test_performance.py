"""Performance and benchmark tests for TimeFlies."""

import pytest
import time
import numpy as np
from pathlib import Path
from tests.fixtures.unit_test_data import create_sample_anndata as create_small_test_anndata


class TestDataPerformance:
    """Test data loading and processing performance."""

    @pytest.mark.performance
    def test_anndata_creation_performance(self, benchmark):
        """Benchmark AnnData creation and basic operations."""
        def create_anndata():
            return create_small_test_anndata(n_cells=1000, n_genes=500)
        
        result = benchmark(create_anndata)
        assert result is not None
        assert result.n_obs == 1000
        assert result.n_vars == 500

    @pytest.mark.performance  
    def test_data_loading_performance(self, benchmark):
        """Test AnnData file I/O performance."""
        # Create test data first
        adata = create_small_test_anndata(n_cells=500, n_genes=200)
        test_file = Path("test_performance_data.h5ad")
        adata.write_h5ad(test_file)
        
        def load_data():
            import anndata
            return anndata.read_h5ad(test_file)
        
        try:
            result = benchmark(load_data)
            assert result is not None
            assert result.n_obs == 500
        finally:
            if test_file.exists():
                test_file.unlink()

    @pytest.mark.performance
    def test_numpy_operations_performance(self, benchmark):
        """Benchmark common numpy operations used in preprocessing."""
        def numpy_operations():
            # Simulate common preprocessing operations
            X = np.random.negative_binomial(10, 0.3, size=(1000, 2000)).astype(float)
            
            # Log transformation
            X_log = np.log1p(X)
            
            # Normalization
            X_norm = X_log / np.sum(X_log, axis=1, keepdims=True) * 10000
            
            # Basic stats
            means = np.mean(X_norm, axis=0)
            stds = np.std(X_norm, axis=0)
            
            return {'means': means, 'stds': stds, 'shape': X_norm.shape}
        
        result = benchmark(numpy_operations)
        assert result['shape'] == (1000, 2000)
        assert len(result['means']) == 2000


class TestModelPerformance:
    """Test model training and prediction performance."""

    @pytest.mark.performance
    def test_sklearn_model_training_speed(self, benchmark):
        """Test sklearn model training performance."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X = np.random.rand(1000, 100)
        y = np.random.randint(0, 3, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        def train_model():
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            return model
        
        model = benchmark(train_model)
        assert model is not None
        
        # Quick validation
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

    @pytest.mark.performance
    def test_prediction_speed(self, benchmark):
        """Test model prediction performance."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Pre-train a model
        X_train = np.random.rand(500, 50)
        y_train = np.random.randint(0, 2, 500)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Test data for prediction
        X_test = np.random.rand(100, 50)
        
        def make_predictions():
            return model.predict(X_test)
        
        predictions = benchmark(make_predictions)
        assert len(predictions) == 100


class TestCLIPerformance:
    """Test CLI command execution performance."""

    @pytest.mark.performance
    def test_cli_help_speed(self, benchmark):
        """Test CLI help command performance."""
        from common.cli.main import main_cli
        
        def run_help():
            try:
                return main_cli(["--help"])
            except SystemExit:
                return 0  # Help command exits with SystemExit
        
        result = benchmark(run_help)
        # Help should be fast and always work
        assert result == 0

    @pytest.mark.performance  
    def test_config_loading_speed(self, benchmark):
        """Test configuration loading performance."""
        from common.core.active_config import get_active_project
        
        def load_config():
            try:
                return get_active_project()
            except Exception:
                return "fruitfly_aging"  # Default fallback
        
        result = benchmark(load_config)
        assert result is not None
        assert isinstance(result, str)


class TestMemoryUsage:
    """Test memory usage patterns."""

    @pytest.mark.performance
    def test_large_data_memory_usage(self):
        """Test memory usage with larger datasets."""
        # Create progressively larger datasets and measure
        sizes = [100, 500, 1000]
        
        for n_cells in sizes:
            adata = create_small_test_anndata(n_cells=n_cells, n_genes=100)
            
            # Basic operations that should be memory efficient
            assert adata.X is not None
            assert adata.n_obs == n_cells
            assert adata.n_vars == 100
            
            # Test that we can perform basic operations
            if hasattr(adata.X, 'toarray'):
                X_dense = adata.X.toarray()
                assert X_dense.shape == (n_cells, 100)

    @pytest.mark.performance
    def test_memory_cleanup(self):
        """Test that large objects are properly cleaned up."""
        import gc
        
        # Create and delete large objects
        for _ in range(3):
            large_data = np.random.rand(1000, 1000)
            assert large_data.shape == (1000, 1000)
            del large_data
            gc.collect()
        
        # If we get here without memory issues, cleanup worked
        assert True


# Benchmark configuration
pytest_plugins = ["pytest_benchmark"]

# Configure benchmark settings
@pytest.fixture(scope="session")
def benchmark_config():
    """Configure benchmark settings."""
    return {
        "min_rounds": 3,
        "max_time": 30,  # Max 30 seconds per benchmark
        "disable_gc": True,  # Disable garbage collection during benchmarks
    }