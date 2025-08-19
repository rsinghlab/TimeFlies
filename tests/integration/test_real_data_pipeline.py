"""Integration tests for TimeFlies pipeline using real data fixtures.

These tests use sampled real data to test the actual pipeline flow,
catching issues that unit tests with mocks might miss.
"""

import pytest
import os
import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.timeflies.core.pipeline_manager import PipelineManager
from src.timeflies.core.config_manager import ConfigManager, Config
from src.timeflies.data.loaders import DataLoader
from src.timeflies.data.preprocessing.data_processor import DataPreprocessor
from src.timeflies.models.model import ModelBuilder


@pytest.fixture
def test_data_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with test fixtures."""
    temp_dir = Path(tempfile.mkdtemp())
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    
    # Create the expected data structure
    data_dir = temp_dir / "data" / "raw" / "h5ad" / "head"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test fixtures to expected locations
    test_files = [
        ("test_data.h5ad", "fly_train.h5ad"),
        ("test_data.h5ad", "fly_eval.h5ad"),
        ("test_data.h5ad", "fly_original.h5ad"),
        ("test_data_corrected.h5ad", "fly_train_batch.h5ad"),
        ("test_data_corrected.h5ad", "fly_eval_batch.h5ad"),
    ]
    
    for src_name, dest_name in test_files:
        src_path = fixtures_dir / src_name
        dest_path = data_dir / dest_name
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def integration_config(temp_data_dir):
    """Create config for integration tests."""
    # Store original working directory
    original_cwd = os.getcwd()
    
    # Change to temp directory for the test
    os.chdir(temp_data_dir)
    
    # Create test configuration dictionary
    config_dict = {
        'general': {
            'project_name': 'TimeFlies',
            'version': '0.2.0',
            'random_state': 42
        },
        'data': {
            'tissue': 'head',
            'model_type': 'CNN',
            'encoding_variable': 'age',
            'cell_type': 'all',
            'sex_type': 'all',
            'batch_correction': {
                'enabled': False
            },
            'filtering': {
                'include_mixed_sex': False
            },
            'sampling': {
                'num_samples': None,
                'num_variables': None
            },
            'train_test_split': {
                'method': 'random',
                'test_split': 0.2,
                'random_state': 42
            }
        },
        'model': {
            'cnn': {
                'filters': [32],
                'kernel_sizes': [3],
                'strides': [1],
                'paddings': ['same'],
                'pool_sizes': [2],
                'pool_strides': [2],
                'dense_units': [64],
                'dropout_rate': 0.3,
                'activation': 'relu'
            },
            'training': {
                'learning_rate': 0.001,
                'epochs': 2,  # Small for testing
                'batch_size': 16,  # Small for testing
                'validation_split': 0.2,
                'early_stopping_patience': 1
            }
        },
        'device': {
            'processor': 'CPU',  # Force CPU for tests
            'gpu_memory_growth': False
        },
        'data_processing': {
            'preprocessing': {
                'required': True,
                'save_data': False  # Don't save during tests
            },
            'model_management': {
                'load_model': False
            },
            'exploratory_data_analysis': {
                'enabled': False
            },
            'normalization': {
                'enabled': False
            }
        },
        'gene_preprocessing': {
            'gene_filtering': {
                'remove_sex_genes': False,
                'remove_autosomal_genes': False,
                'only_keep_lnc_genes': False,
                'remove_lnc_genes': False,
                'remove_unaccounted_genes': False,
                'select_batch_genes': False,
                'highly_variable_genes': False
            },
            'gene_balancing': {
                'balance_genes': False,
                'balance_lnc_genes': False
            },
            'gene_shuffle': {
                'shuffle_genes': False
            }
        },
        'feature_importance': {
            'run_interpreter': False,  # Skip SHAP for speed
            'run_visualization': False  # Skip viz for speed
        },
        'file_locations': {
            'training_file': 'fly_train.h5ad',
            'evaluation_file': 'fly_eval.h5ad',
            'original_file': 'fly_original.h5ad',
            'batch_corrected_files': {
                'train': 'fly_train_batch.h5ad',
                'eval': 'fly_eval_batch.h5ad'
            }
        }
    }
    
    # Create Config object from dictionary
    test_config = Config(config_dict)
    
    yield test_config
    
    # Restore original working directory
    os.chdir(original_cwd)


class TestRealDataPipelineIntegration:
    """Test the complete pipeline using real data fixtures."""
    
    @pytest.mark.integration
    def test_data_loading_with_real_fixtures(self, integration_config, temp_data_dir):
        """Test that data loading works with real fixture data."""
        pipeline = PipelineManager(integration_config)
        
        # Test data loading
        pipeline.load_data()
        
        # Verify data was loaded
        assert pipeline.adata is not None
        assert pipeline.adata_eval is not None
        assert pipeline.adata_original is not None
        
        # Check data dimensions match our test fixtures
        assert pipeline.adata.n_obs == 100
        assert pipeline.adata.n_vars == 500
        
        # Check that required metadata columns exist
        required_cols = ['age', 'sex', 'tissue', 'dataset']
        for col in required_cols:
            assert col in pipeline.adata.obs.columns
            
        print(f"✓ Data loading test passed - loaded {pipeline.adata.shape} data")
    
    @pytest.mark.integration
    def test_data_preprocessing_with_real_fixtures(self, integration_config, temp_data_dir):
        """Test data preprocessing with real fixture data."""
        pipeline = PipelineManager(integration_config)
        
        # Load and preprocess data
        pipeline.load_data()
        pipeline.preprocess_data()
        
        # Verify preprocessing results
        assert pipeline.train_data is not None
        assert pipeline.test_data is not None
        assert pipeline.train_labels is not None  
        assert pipeline.test_labels is not None
        assert pipeline.label_encoder is not None
        
        # Check data shapes are reasonable
        assert len(pipeline.train_data.shape) >= 2
        assert len(pipeline.test_data.shape) >= 2
        assert pipeline.train_data.shape[0] > 0
        assert pipeline.test_data.shape[0] > 0
        
        # For CNN, data should be 3D
        if integration_config.data.model_type == "CNN":
            assert len(pipeline.train_data.shape) == 3
            assert len(pipeline.test_data.shape) == 3
            assert pipeline.train_data.shape[1] == 1  # Time dimension
            
        print(f"✓ Data preprocessing test passed - train: {pipeline.train_data.shape}, test: {pipeline.test_data.shape}")
    
    @pytest.mark.integration 
    def test_model_building_with_real_fixtures(self, integration_config, temp_data_dir):
        """Test model building with real fixture data."""
        pipeline = PipelineManager(integration_config)
        
        # Load and preprocess data
        pipeline.load_data()
        pipeline.preprocess_data()
        
        # Build model
        pipeline.build_and_train_model()
        
        # Verify model was created
        assert pipeline.model is not None
        assert pipeline.history is not None
        
        # Check that model can make predictions
        predictions = pipeline.model.predict(pipeline.test_data)
        assert predictions is not None
        assert predictions.shape[0] == pipeline.test_data.shape[0]
        
        print(f"✓ Model building test passed - model trained and can predict")
    
    @pytest.mark.integration
    def test_batch_correction_disabled_uses_correct_data(self, integration_config, temp_data_dir):
        """Test that batch correction disabled actually uses uncorrected data."""
        # Ensure batch correction is disabled
        integration_config.data.batch_correction.enabled = False
        
        pipeline = PipelineManager(integration_config)
        pipeline.load_data()
        
        # Create data processor to test correct data usage
        processor = DataPreprocessor(
            integration_config,
            pipeline.adata,
            pipeline.adata_corrected
        )
        
        # Test prepare_data method
        result = processor.prepare_data()
        train_data, test_data, train_labels, test_labels, label_encoder, reference_data, scaler, is_scaler_fit, highly_variable_genes, mix_included = result
        
        # With batch correction disabled, should use uncorrected data
        # The uncorrected test data has sparsity ~0.966, corrected would be different
        assert train_data is not None
        assert test_data is not None
        
        # Check that we're using the right dataset by examining sparsity
        sparsity = float(np.mean(train_data == 0))
        assert sparsity > 0.9  # High sparsity indicates uncorrected data
        
        print(f"✓ Batch correction disabled test passed - using uncorrected data (sparsity: {sparsity:.3f})")
    
    @pytest.mark.integration
    def test_cnn_reshaping_works_correctly(self, integration_config, temp_data_dir):
        """Test that CNN data reshaping works without shape errors."""
        # Force CNN model type
        integration_config.data.model_type = "CNN"
        
        pipeline = PipelineManager(integration_config)
        pipeline.load_data()
        pipeline.preprocess_data()
        
        # Verify CNN reshaping worked
        assert len(pipeline.train_data.shape) == 3
        assert pipeline.train_data.shape[1] == 1  # Should be reshaped to (samples, 1, features)
        
        # Try to build a CNN model - this should not fail with shape errors
        builder = ModelBuilder(
            integration_config,
            pipeline.train_data,
            pipeline.train_labels,
            pipeline.label_encoder,
            pipeline.reference_data,
            pipeline.scaler,
            pipeline.is_scaler_fit,
            pipeline.highly_variable_genes,
            pipeline.mix_included
        )
        
        model = builder.create_cnn_model(num_output_units=len(np.unique(pipeline.train_labels.argmax(axis=1))))
        assert model is not None
        
        # Test that model accepts the reshaped data
        predictions = model.predict(pipeline.train_data[:10])  # Just test first 10 samples
        assert predictions is not None
        
        print(f"✓ CNN reshaping test passed - data shape: {pipeline.train_data.shape}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_pipeline_no_batch_correction(self, integration_config, temp_data_dir):
        """Test complete pipeline end-to-end without batch correction."""
        # Disable batch correction
        integration_config.data.batch_correction.enabled = False
        
        pipeline = PipelineManager(integration_config)
        
        # Run complete pipeline
        pipeline.load_data()
        pipeline.preprocess_data()
        pipeline.build_and_train_model()
        
        # Verify everything completed successfully
        assert pipeline.adata is not None
        assert pipeline.train_data is not None
        assert pipeline.model is not None
        assert pipeline.history is not None
        
        # Test final predictions
        predictions = pipeline.model.predict(pipeline.test_data)
        assert predictions.shape[0] == pipeline.test_data.shape[0]
        
        # Check that history contains training metrics
        assert 'loss' in pipeline.history.history
        
        print("✓ End-to-end pipeline test passed - complete pipeline executed successfully")


@pytest.mark.integration
class TestDataConsistency:
    """Test that data flow is consistent and memory-efficient."""
    
    def test_memory_usage_without_batch_correction(self, integration_config, temp_data_dir):
        """Test that disabling batch correction doesn't load corrected data."""
        integration_config.data.batch_correction.enabled = False
        
        pipeline = PipelineManager(integration_config)
        pipeline.load_data()
        
        # When batch correction is disabled, corrected data should still be loaded
        # but not used in preprocessing
        assert pipeline.adata_corrected is not None  # Still loaded
        
        # But preprocessing should only use uncorrected data
        processor = DataPreprocessor(
            integration_config,
            pipeline.adata,
            pipeline.adata_corrected
        )
        
        # Mock the process_adata method to track which data is processed
        processed_data = []
        original_process_adata = processor.process_adata
        
        def track_process_adata(adata):
            processed_data.append(id(adata))
            return original_process_adata(adata)
        
        processor.process_adata = track_process_adata
        processor.prepare_data()
        
        # Should only process one dataset (the uncorrected one)
        assert len(processed_data) == 1
        # Should be processing the uncorrected data
        assert processed_data[0] == id(pipeline.adata)
        
        print("✓ Memory usage test passed - only uncorrected data processed when batch correction disabled")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])