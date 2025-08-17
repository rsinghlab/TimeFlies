"""Integration tests for end-to-end pipeline functionality."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch
from anndata import AnnData

from src.timeflies.core.config_manager import ConfigManager
from src.timeflies.core.pipeline_manager import PipelineManager


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_data = self.create_test_config()
        
    def create_test_config(self):
        """Create test configuration data."""
        return {
            'data': {
                'tissue': 'head',
                'sex_type': 'all',
                'cell_type': 'all',
                'encoding_variable': 'age',
                'model_type': 'CNN',
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
                    'random_state': 42,
                    'train': {
                        'sex': 'male',
                        'tissue': 'head'
                    },
                    'test': {
                        'sex': 'female',
                        'tissue': 'body',
                        'size': 0.3
                    }
                },
                'batch_correction': {
                    'enabled': False
                }
            },
            'gene_preprocessing': {
                'gene_shuffle': {
                    'shuffle_genes': False,
                    'shuffle_random_state': 42
                },
                'gene_filtering': {
                    'highly_variable_genes': False,
                    'select_batch_genes': False
                }
            },
            'data_processing': {
                'normalization': {
                    'enabled': False
                },
                'exploratory_data_analysis': {
                    'enabled': False
                },
                'model_management': {
                    'load_model': False
                }
            },
            'model': {
                'cnn': {
                    'filters': [64, 128],
                    'kernel_sizes': [3, 3],
                    'strides': [1, 1],
                    'paddings': ['same', 'same'],
                    'pool_sizes': [2, None],
                    'pool_strides': [2, 1],
                    'dense_units': [64, 32],
                    'activation_function': 'relu',
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001
                },
                'training': {
                    'custom_loss': 'sparse_categorical_crossentropy',
                    'metrics': ['accuracy'],
                    'early_stopping_patience': 5,
                    'epochs': 10,
                    'batch_size': 32,
                    'validation_split': 0.2
                }
            },
            'device': {
                'processor': 'GPU'
            },
            'feature_importance': {
                'reference_size': 50,
                'load_SHAP': False,
                'run_interpreter': False,
                'run_visualization': False,
                'save_predictions': False
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
        
    def create_sample_adata(self, n_obs=1000, n_vars=100):
        """Create sample AnnData object for testing."""
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        obs = pd.DataFrame({
            'age': np.random.choice([1, 5, 10, 20], n_obs),
            'sex': np.random.choice(['male', 'female'], n_obs),
            'tissue': np.random.choice(['head', 'body'], n_obs),
            'afca_annotation_broad': np.random.choice([
                'CNS neuron', 'muscle cell', 'epithelial cell'
            ], n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        var = pd.DataFrame({
            'gene_type': np.random.choice(['protein_coding', 'lncRNA'], n_vars),
            'highly_variable': np.random.choice([True, False], n_vars, p=[0.2, 0.8])
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        return AnnData(X=X, obs=obs, var=var)
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    def test_pipeline_initialization_and_setup(self, mock_path_manager, mock_data_loader, mock_gpu_handler):
        """Test pipeline initialization and basic setup."""
        config = ConfigManager.from_dict(self.config_data)
        pipeline = PipelineManager(config)
        
        # Test GPU setup
        pipeline.setup_gpu()
        mock_gpu_handler.configure.assert_called_once_with(config)
        
        # Test data loading setup
        mock_loader_instance = mock_data_loader.return_value
        mock_loader_instance.load_data.return_value = (
            self.create_sample_adata(),
            self.create_sample_adata(500),
            self.create_sample_adata()
        )
        mock_loader_instance.load_corrected_data.return_value = (
            self.create_sample_adata(),
            self.create_sample_adata(500)
        )
        mock_loader_instance.load_gene_lists.return_value = (
            [f'autosomal_gene_{i}' for i in range(50)],
            [f'sex_gene_{i}' for i in range(10)]
        )
        
        pipeline.load_data()
        
        # Verify data loading was called
        mock_loader_instance.load_data.assert_called_once()
        mock_loader_instance.load_corrected_data.assert_called_once()
        mock_loader_instance.load_gene_lists.assert_called_once()
        
        # Verify attributes are set
        assert hasattr(pipeline, 'adata')
        assert hasattr(pipeline, 'adata_eval')
        assert hasattr(pipeline, 'adata_original')
        assert hasattr(pipeline, 'adata_corrected')
        assert hasattr(pipeline, 'adata_eval_corrected')
        assert hasattr(pipeline, 'autosomal_genes')
        assert hasattr(pipeline, 'sex_genes')
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    @patch('src.timeflies.data.preprocessing.gene_filter.GeneFilter')
    @patch('src.timeflies.data.preprocessing.data_processor.DataPreprocessor')
    def test_preprocessing_pipeline(self, mock_data_processor, mock_gene_filter, 
                                   mock_path_manager, mock_data_loader, mock_gpu_handler):
        """Test data preprocessing pipeline."""
        config = ConfigManager.from_dict(self.config_data)
        pipeline = PipelineManager(config)
        
        # Set up mock data
        pipeline.adata = self.create_sample_adata()
        pipeline.adata_eval = self.create_sample_adata(500)
        pipeline.adata_original = self.create_sample_adata()
        pipeline.adata_corrected = self.create_sample_adata()
        pipeline.autosomal_genes = [f'autosomal_gene_{i}' for i in range(50)]
        pipeline.sex_genes = [f'sex_gene_{i}' for i in range(10)]
        
        # Mock gene filter
        mock_filter_instance = mock_gene_filter.return_value
        mock_filter_instance.apply_filter.return_value = (
            pipeline.adata, pipeline.adata_eval, pipeline.adata_original
        )
        
        # Mock data preprocessor
        mock_processor_instance = mock_data_processor.return_value
        mock_processor_instance.prepare_data.return_value = (
            np.random.randn(800, 100),  # train_data
            np.random.randn(200, 100),  # test_data
            np.eye(3)[np.random.choice(3, 800)],  # train_labels
            np.eye(3)[np.random.choice(3, 200)],  # test_labels
            Mock(),  # label_encoder
            np.random.randn(50, 100),  # reference_data
            Mock(),  # scaler
            True,  # is_scaler_fit
            ['gene1', 'gene2'],  # highly_variable_genes
            False  # mix_included
        )
        
        # Test gene filtering
        pipeline.setup_gene_filtering()
        mock_gene_filter.assert_called_once()
        mock_filter_instance.apply_filter.assert_called_once()
        
        # Test data preprocessing
        pipeline.preprocess_data()
        mock_data_processor.assert_called_once()
        mock_processor_instance.prepare_data.assert_called_once()
        
        # Verify preprocessing results
        assert hasattr(pipeline, 'train_data')
        assert hasattr(pipeline, 'test_data')
        assert hasattr(pipeline, 'train_labels')
        assert hasattr(pipeline, 'test_labels')
        assert hasattr(pipeline, 'label_encoder')
        assert hasattr(pipeline, 'reference_data')
        assert hasattr(pipeline, 'scaler')
        assert hasattr(pipeline, 'is_scaler_fit')
        assert hasattr(pipeline, 'highly_variable_genes')
        assert hasattr(pipeline, 'mix_included')
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    @patch('src.timeflies.models.model.ModelBuilder')
    def test_model_building_pipeline(self, mock_model_builder, mock_path_manager, 
                                    mock_data_loader, mock_gpu_handler):
        """Test model building and training pipeline."""
        config = ConfigManager.from_dict(self.config_data)
        pipeline = PipelineManager(config)
        
        # Set up required attributes
        pipeline.train_data = np.random.randn(800, 1, 100)  # CNN format
        pipeline.train_labels = np.eye(3)[np.random.choice(3, 800)]
        pipeline.label_encoder = Mock()
        pipeline.reference_data = np.random.randn(50, 100)
        pipeline.scaler = Mock()
        pipeline.is_scaler_fit = True
        pipeline.highly_variable_genes = ['gene1', 'gene2']
        pipeline.mix_included = False
        
        # Mock model builder
        mock_builder_instance = mock_model_builder.return_value
        mock_model = Mock()
        mock_history = Mock()
        mock_history.history = {'loss': [0.8, 0.6, 0.4], 'accuracy': [0.6, 0.7, 0.8]}
        mock_builder_instance.run.return_value = (mock_model, mock_history)
        
        # Test model building
        pipeline.build_and_train_model()
        
        mock_model_builder.assert_called_once_with(
            config, pipeline.train_data, pipeline.train_labels,
            pipeline.label_encoder, pipeline.reference_data,
            pipeline.scaler, pipeline.is_scaler_fit,
            pipeline.highly_variable_genes, pipeline.mix_included
        )
        mock_builder_instance.run.assert_called_once()
        
        # Verify model and history are set
        assert pipeline.model == mock_model
        assert pipeline.history == mock_history
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    @patch('src.timeflies.models.model.ModelLoader')
    def test_model_loading_pipeline(self, mock_model_loader, mock_path_manager, 
                                   mock_data_loader, mock_gpu_handler):
        """Test model loading pipeline."""
        # Enable model loading in config
        config_data = self.config_data.copy()
        config_data['data_processing']['model_management']['load_model'] = True
        
        config = ConfigManager.from_dict(config_data)
        pipeline = PipelineManager(config)
        
        # Mock model loader
        mock_loader_instance = mock_model_loader.return_value
        mock_loader_instance.load_model_components.return_value = (
            Mock(),  # label_encoder
            Mock(),  # scaler
            True,    # is_scaler_fit
            ['gene1', 'gene2'],  # highly_variable_genes
            100,     # num_features
            Mock(),  # history
            False,   # mix_included
            np.random.randn(50, 100)  # reference_data
        )
        mock_loader_instance.load_model.return_value = Mock()
        
        # Test model loading
        pipeline.load_model_components()
        pipeline.load_model()
        
        mock_model_loader.assert_called_once_with(config)
        mock_loader_instance.load_model_components.assert_called_once()
        mock_loader_instance.load_model.assert_called_once()
        
        # Verify components are loaded
        assert hasattr(pipeline, 'label_encoder')
        assert hasattr(pipeline, 'scaler')
        assert hasattr(pipeline, 'is_scaler_fit')
        assert hasattr(pipeline, 'highly_variable_genes')
        assert hasattr(pipeline, 'num_features')
        assert hasattr(pipeline, 'history')
        assert hasattr(pipeline, 'mix_included')
        assert hasattr(pipeline, 'reference_data')
        assert hasattr(pipeline, 'model')
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    @patch('src.timeflies.evaluation.interpreter.Metrics')
    def test_metrics_computation_pipeline(self, mock_metrics, mock_path_manager, 
                                         mock_data_loader, mock_gpu_handler):
        """Test metrics computation pipeline."""
        config = ConfigManager.from_dict(self.config_data)
        pipeline = PipelineManager(config)
        
        # Set up required attributes
        pipeline.model = Mock()
        pipeline.test_data = np.random.randn(200, 100)
        pipeline.test_labels = np.eye(3)[np.random.choice(3, 200)]
        pipeline.label_encoder = Mock()
        pipeline.path_manager = Mock()
        
        # Mock metrics
        mock_metrics_instance = mock_metrics.return_value
        
        # Test metrics computation
        pipeline.run_metrics()
        
        mock_metrics.assert_called_once_with(
            config, pipeline.model, pipeline.test_data,
            pipeline.test_labels, pipeline.label_encoder, pipeline.path_manager
        )
        mock_metrics_instance.compute_metrics.assert_called_once()
        
    def test_config_integration(self):
        """Test configuration integration with ConfigManager."""
        config = ConfigManager.from_dict(self.config_data)
        
        # Test accessing nested configuration values
        assert config.data.model_type == 'CNN'
        assert config.data.encoding_variable == 'age'
        assert config.model.cnn.filters == [64, 128]
        assert config.model.training.epochs == 10
        assert config.data.batch_correction.enabled == False
        assert config.feature_importance.run_interpreter == False
        
        # Test getattr functionality for missing attributes
        assert hasattr(config.data, 'nonexistent_attr') == False
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    def test_pipeline_error_handling(self, mock_path_manager, mock_data_loader, mock_gpu_handler):
        """Test pipeline error handling."""
        config = ConfigManager.from_dict(self.config_data)
        pipeline = PipelineManager(config)
        
        # Test GPU setup failure
        mock_gpu_handler.configure.side_effect = Exception("GPU setup failed")
        with pytest.raises(Exception, match="GPU setup failed"):
            pipeline.setup_gpu()
            
        # Test data loading failure
        mock_loader_instance = mock_data_loader.return_value
        mock_loader_instance.load_data.side_effect = Exception("Data loading failed")
        with pytest.raises(Exception, match="Data loading failed"):
            pipeline.load_data()
            
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test with minimal configuration
        minimal_config = {
            'data': {
                'tissue': 'head'
            }
        }
        
        config = ConfigManager.from_dict(minimal_config)
        
        # Test that we can access the configuration without errors
        assert config.data.tissue == 'head'
        
        # Test accessing non-existent attributes doesn't raise errors
        # (should use getattr with defaults in the actual code)
        tissue = getattr(config.data, 'tissue', 'default_tissue')
        assert tissue == 'head'
        
        encoding_var = getattr(config.data, 'encoding_variable', 'age')
        assert encoding_var == 'age'  # Should use default
        
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    @patch('src.timeflies.data.loaders.DataLoader')
    @patch('src.timeflies.utils.path_manager.PathManager')
    def test_conditional_pipeline_execution(self, mock_path_manager, mock_data_loader, mock_gpu_handler):
        """Test conditional execution of pipeline components."""
        # Test with EDA enabled
        config_data = self.config_data.copy()
        config_data['data_processing']['exploratory_data_analysis']['enabled'] = True
        config_data['feature_importance']['run_interpreter'] = True
        config_data['feature_importance']['run_visualization'] = True
        
        config = ConfigManager.from_dict(config_data)
        pipeline = PipelineManager(config)
        
        # Set up required data
        pipeline.adata = self.create_sample_adata()
        pipeline.adata_eval = self.create_sample_adata(500)
        pipeline.adata_original = self.create_sample_adata()
        pipeline.adata_corrected = self.create_sample_adata()
        pipeline.adata_eval_corrected = self.create_sample_adata(500)
        
        with patch('src.timeflies.analysis.eda.EDAHandler') as mock_eda:
            mock_eda_instance = mock_eda.return_value
            pipeline.run_eda()
            
            # EDA should be called when enabled
            mock_eda.assert_called_once()
            mock_eda_instance.run_eda.assert_called_once()
            
        # Test with interpretation enabled
        pipeline.model = Mock()
        pipeline.test_data = np.random.randn(200, 100)
        pipeline.test_labels = np.eye(3)[np.random.choice(3, 200)]
        pipeline.label_encoder = Mock()
        pipeline.reference_data = np.random.randn(50, 100)
        pipeline.path_manager = Mock()
        
        with patch('src.timeflies.evaluation.interpreter.Interpreter') as mock_interpreter:
            mock_interpreter_instance = mock_interpreter.return_value
            mock_interpreter_instance.compute_or_load_shap_values.return_value = (Mock(), Mock())
            
            pipeline.run_interpretation()
            
            # Interpreter should be called when enabled
            mock_interpreter.assert_called_once()
            mock_interpreter_instance.compute_or_load_shap_values.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])