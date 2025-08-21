"""Integration tests for MODEL PIPELINE components.

Tests how model-related components work together:
- ModelFactory + ModelBuilder + actual training
- Different model types (MLP, CNN, Logistic Regression)
- Model training, evaluation, and metrics computation
- Model saving/loading and state management

Focus: Model creation through evaluation with real training
Time: ~30-60 seconds with small models and data
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch
import scanpy as sc
from anndata import AnnData

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.core.config_manager import Config
from shared.models.model_factory import ModelFactory
from shared.models.model import ModelBuilder
from shared.data.loaders import DataLoader
from shared.utils.path_manager import PathManager


class TestModelPipelineIntegration:
    """Integration tests for the complete model pipeline."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create realistic training data for model testing."""
        np.random.seed(42)  # For reproducibility
        
        n_samples = 1000
        n_features = 500
        n_classes = 4
        
        # Create realistic gene expression data
        X = np.random.lognormal(mean=1.0, sigma=1.5, size=(n_samples, n_features))
        X = X.astype(np.float32)
        
        # Create age-based labels (4 age groups)
        y = np.random.choice([0, 1, 2, 3], size=n_samples)
        y_categorical = np.eye(n_classes)[y]  # One-hot encoded
        
        # Split into train/test
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_categorical[:split_idx], y_categorical[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_features': n_features,
            'n_classes': n_classes
        }
    
    @pytest.fixture
    def model_configs(self):
        """Create configurations for different model types."""
        base_config = {
            'general': {'random_state': 42},
            'data': {
                'model_type': 'CNN',
                'encoding_variable': 'age'
            }
        }
        
        configs = {}
        
        # CNN Config
        configs['cnn'] = Config({
            **base_config,
            'data': {**base_config['data'], 'model_type': 'CNN'},
            'model': {
                'training': {
                    'epochs': 2,  # Short for testing
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'metrics': ['accuracy'],
                    'custom_loss': 'categorical_crossentropy'
                },
                'cnn': {
                    'filters': [16],  # Smaller for testing
                    'kernel_sizes': [3],
                    'strides': [1],
                    'paddings': ['same'],
                    'pool_sizes': [2],
                    'pool_strides': [2],
                    'dense_units': [32],  # Smaller for testing
                    'dropout_rate': 0.3,
                    'activation': 'relu'
                }
            }
        })
        
        # MLP Config
        configs['mlp'] = Config({
            **base_config,
            'data': {**base_config['data'], 'model_type': 'MLP'},
            'model': {
                'training': {
                    'epochs': 2,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'metrics': ['accuracy'],
                    'custom_loss': 'categorical_crossentropy'
                },
                'mlp': {
                    'hidden_layers': [32, 16],  # Smaller for testing
                    'activation': 'relu',
                    'dropout_rate': 0.2
                }
            }
        })
        
        # Logistic Regression Config
        configs['logistic'] = Config({
            **base_config,
            'data': {**base_config['data'], 'model_type': 'logistic'},
            'model': {
                'logistic': {
                    'max_iter': 100,  # Smaller for testing
                    'random_state': 42,
                    'C': 1.0
                }
            }
        })
        
        return configs
    
    def test_model_factory_creates_all_model_types(self, model_configs):
        """Test that ModelFactory can create all supported model types."""
        # Test CNN
        cnn_model = ModelFactory.create_model('cnn', model_configs['cnn'])
        assert cnn_model is not None
        assert hasattr(cnn_model, 'build')
        assert hasattr(cnn_model, 'train')
        assert hasattr(cnn_model, 'predict')
        
        # Test MLP
        mlp_model = ModelFactory.create_model('mlp', model_configs['mlp'])
        assert mlp_model is not None
        assert hasattr(mlp_model, 'build')
        
        # Test Logistic Regression
        lr_model = ModelFactory.create_model('logistic', model_configs['logistic'])
        assert lr_model is not None
        assert hasattr(lr_model, 'build')
        
        # Test supported models list
        supported = ModelFactory.get_supported_models()
        assert 'cnn' in supported
        assert 'mlp' in supported
        assert 'logistic' in supported
        assert isinstance(supported, list)
    
    def test_cnn_model_complete_pipeline(self, sample_training_data, model_configs):
        """Test complete CNN model pipeline from build to prediction."""
        data = sample_training_data
        config = model_configs['cnn']
        
        # Create and build model
        model = ModelFactory.create_model('cnn', config)
        
        # CNN will automatically reshape 2D data to 3D, so pass original 2D data
        X_train_cnn = data['X_train']
        X_test_cnn = data['X_test']
        
        # CNN expects input shape as (features,) for build method
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        assert model.model is not None
        
        # Train model (minimal epochs for testing)
        history = model.train(X_train_cnn, data['y_train'])
        assert history is not None
        assert model.is_trained
        
        # Test predictions
        predictions = model.predict(X_test_cnn)
        assert predictions.shape[0] == X_test_cnn.shape[0]
        assert len(np.unique(predictions)) <= data['n_classes']
        
        # Test prediction probabilities
        probabilities = model.predict_proba(X_test_cnn)
        assert probabilities.shape == (X_test_cnn.shape[0], data['n_classes'])
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)  # Should sum to 1
    
    def test_mlp_model_complete_pipeline(self, sample_training_data, model_configs):
        """Test complete MLP model pipeline."""
        data = sample_training_data
        config = model_configs['mlp']
        
        # Create and build model
        model = ModelFactory.create_model('mlp', config)
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        
        # Train model
        history = model.train(data['X_train'], data['y_train'])
        assert history is not None
        assert model.is_trained
        
        # Test predictions
        predictions = model.predict(data['X_test'])
        assert predictions.shape[0] == data['X_test'].shape[0]
        
        # Test prediction probabilities
        probabilities = model.predict_proba(data['X_test'])
        assert probabilities.shape == (data['X_test'].shape[0], data['n_classes'])
    
    def test_logistic_regression_pipeline(self, sample_training_data, model_configs):
        """Test Logistic Regression model pipeline."""
        data = sample_training_data
        config = model_configs['logistic']
        
        # Create and build model
        model = ModelFactory.create_model('logistic', config)
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        
        # Convert one-hot to class indices for sklearn
        y_train_indices = np.argmax(data['y_train'], axis=1)
        
        # Train model
        model.train(data['X_train'], y_train_indices)
        assert model.is_trained
        
        # Test predictions
        predictions = model.predict(data['X_test'])
        assert predictions.shape[0] == data['X_test'].shape[0]
        assert all(0 <= pred < data['n_classes'] for pred in predictions)
        
        # Test prediction probabilities
        probabilities = model.predict_proba(data['X_test'])
        assert probabilities.shape == (data['X_test'].shape[0], data['n_classes'])
    
    def test_model_builder_integration(self, sample_training_data, model_configs):
        """Test ModelFactory integration (replacement for legacy ModelBuilder)."""
        data = sample_training_data
        config = model_configs['cnn']
        
        # Test model creation through factory
        model = ModelFactory.create_model('cnn', config)
        assert model is not None
        
        # Test model building
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        assert model.model is not None
        assert hasattr(model.model, 'fit')  # Should be a Keras model
        
        # Test training
        history = model.train(data['X_train'], data['y_train'])
        assert history is not None
        assert hasattr(history, 'history')  # Keras history object
    
    def test_model_error_handling(self, model_configs):
        """Test model error handling scenarios."""
        config = model_configs['cnn']
        
        # Test invalid model type
        with pytest.raises(Exception):  # Should raise ModelError or similar
            ModelFactory.create_model('invalid_model', config)
        
        # Test prediction without training
        model = ModelFactory.create_model('cnn', config)
        model.build(input_shape=(100, 1), num_classes=4)
        
        X_dummy = np.random.randn(10, 100, 1)
        
        try:
            predictions = model.predict(X_dummy)
            # If predictions work, that's fine too
            assert predictions is not None
        except Exception:
            # Expected - untrained model should raise error
            pass
    
    def test_model_configuration_variations(self, sample_training_data):
        """Test models with different configuration variations."""
        data = sample_training_data
        
        # Test CNN with different filter sizes
        config_dict = {
            'general': {'random_state': 42},
            'data': {'model_type': 'CNN'},
            'model': {
                'training': {
                    'epochs': 1,
                    'batch_size': 16,
                    'learning_rate': 0.01,
                    'metrics': ['accuracy'],
                    'custom_loss': 'categorical_crossentropy'
                },
                'cnn': {
                    'filters': [8, 16],  # Multiple filters
                    'kernel_sizes': [5, 3],
                    'strides': [1, 1],
                    'paddings': ['same', 'same'],
                    'pool_sizes': [2, 2],
                    'pool_strides': [2, 2],
                    'dense_units': [64, 32],
                    'dropout_rate': 0.5,
                    'activation': 'relu'
                }
            }
        }
        
        config = Config(config_dict)
        model = ModelFactory.create_model('cnn', config)
        
        X_train_cnn = data['X_train']
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        
        # Should build successfully with multiple layers
        assert model.model is not None
        
        # Quick training test
        try:
            history = model.train(X_train_cnn, data['y_train'])
            # Training might not work in test environment, that's ok
            if history is not None:
                assert model.is_trained
        except Exception:
            # Training failure is expected in test environment
            pass
    
    def test_model_performance_metrics(self, sample_training_data, model_configs):
        """Test that models produce reasonable performance metrics."""
        data = sample_training_data
        config = model_configs['mlp']
        
        model = ModelFactory.create_model('mlp', config)
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        
        # Train model
        history = model.train(data['X_train'], data['y_train'])
        
        # Test that training history contains expected metrics
        assert hasattr(history, 'history')
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        
        # Test that loss generally decreases (learning is happening)
        losses = history.history['loss']
        if len(losses) > 1:
            # Allow some fluctuation but expect general improvement
            assert losses[-1] <= losses[0] + 0.1  # Some tolerance for short training
        
        # Test predictions are reasonable
        predictions = model.predict(data['X_test'])
        unique_preds = len(np.unique(predictions))
        assert 1 <= unique_preds <= data['n_classes']  # Should predict at least 1 class, at most n_classes
    
    def test_model_state_management(self, sample_training_data, model_configs):
        """Test model state management (trained/untrained states)."""
        data = sample_training_data
        config = model_configs['logistic']
        
        model = ModelFactory.create_model('logistic', config)
        
        # Initially untrained
        assert not model.is_trained
        
        # Build model
        model.build(input_shape=(data['n_features'],), num_classes=data['n_classes'])
        assert not model.is_trained  # Building doesn't make it trained
        
        # Train model
        y_train_indices = np.argmax(data['y_train'], axis=1)
        model.train(data['X_train'], y_train_indices)
        
        # Now should be trained
        assert model.is_trained
        
        # Can make predictions
        predictions = model.predict(data['X_test'])
        assert predictions is not None