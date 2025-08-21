"""Unit tests for PipelineManager - testing core functionality."""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.core.pipeline_manager import PipelineManager
from shared.core.config_manager import Config


class TestPipelineManagerCore:
    """Test core PipelineManager functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config_dict = {
            'general': {'random_state': 42},
            'device': {'processor': 'GPU'},
            'data': {
                'tissue': 'head',
                'encoding_variable': 'age',
                'model_type': 'cnn',
                'batch_correction': {'enabled': False},
                'filtering': {'include_mixed_sex': False},
                'train_test_split': {
                    'test_split': 0.2,
                    'random_state': 42
                }
            },
            'data_processing': {
                'exploratory_data_analysis': {'enabled': False},
                'model_management': {'load_model': False}
            },
            'gene_preprocessing': {
                'gene_filtering': {
                    'highly_variable_genes': False,
                    'remove_sex_genes': False,
                    'remove_autosomal_genes': False,
                    'select_batch_genes': False
                }
            },
            'feature_importance': {
                'run_interpreter': False,
                'run_visualization': False
            }
        }
        return Config(config_dict)
    
    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData for testing."""
        n_obs, n_vars = 100, 50
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        obs = pd.DataFrame({
            'age': np.random.choice([5, 30, 50, 70], n_obs),
            'sex': np.random.choice(['male', 'female'], n_obs),
            'tissue': 'head',
            'batch': np.random.choice(['batch1', 'batch2'], n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        var = pd.DataFrame({
            'gene_symbol': [f'gene_{i}' for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], n_vars, p=[0.3, 0.7])
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        return AnnData(X=X, obs=obs, var=var)
    
    def test_pipeline_manager_initialization(self, mock_config):
        """Test PipelineManager initialization."""
        with patch('shared.core.pipeline_manager.DataLoader') as mock_data_loader:
            with patch('shared.core.pipeline_manager.PathManager') as mock_path_manager:
                pipeline = PipelineManager(mock_config)
                
                assert pipeline.config_instance == mock_config
                mock_data_loader.assert_called_once_with(mock_config)
                mock_path_manager.assert_called_once_with(mock_config)
    
    @patch('shared.core.pipeline_manager.GPUHandler')
    def test_setup_gpu_success(self, mock_gpu_handler, mock_config):
        """Test successful GPU setup."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Mock successful GPU configuration
                mock_gpu_handler.configure.return_value = None
                
                pipeline.setup_gpu()
                
                mock_gpu_handler.configure.assert_called_once_with(mock_config)
    
    @patch('shared.core.pipeline_manager.GPUHandler')
    def test_setup_gpu_failure(self, mock_gpu_handler, mock_config):
        """Test GPU setup failure."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Mock GPU configuration failure
                mock_gpu_handler.configure.side_effect = Exception("GPU error")
                
                with pytest.raises(Exception, match="GPU error"):
                    pipeline.setup_gpu()
    
    def test_load_data_no_batch_correction(self, mock_config, sample_adata):
        """Test data loading without batch correction."""
        mock_config.data.batch_correction.enabled = False
        
        with patch('shared.core.pipeline_manager.DataLoader') as mock_data_loader_class:
            with patch('shared.core.pipeline_manager.PathManager'):
                # Mock data loader instance
                mock_data_loader = Mock()
                mock_data_loader_class.return_value = mock_data_loader
                
                # Mock data loading methods
                mock_data_loader.load_data.return_value = (sample_adata, sample_adata.copy(), sample_adata.copy())
                mock_data_loader.load_gene_lists.return_value = (['gene1', 'gene2'], ['sexgene1'])
                
                pipeline = PipelineManager(mock_config)
                pipeline.load_data()
                
                # Should load basic data
                mock_data_loader.load_data.assert_called_once()
                mock_data_loader.load_gene_lists.assert_called_once()
                
                # Should not load corrected data
                assert not hasattr(mock_data_loader, 'load_corrected_data') or not mock_data_loader.load_corrected_data.called
                
                # Check attributes
                assert hasattr(pipeline, 'adata')
                assert hasattr(pipeline, 'adata_eval')
                assert hasattr(pipeline, 'adata_original')
                assert pipeline.adata_corrected is None
                assert pipeline.adata_eval_corrected is None
    
    def test_load_data_with_batch_correction(self, mock_config, sample_adata):
        """Test data loading with batch correction."""
        mock_config.data.batch_correction.enabled = True
        
        with patch('shared.core.pipeline_manager.DataLoader') as mock_data_loader_class:
            with patch('shared.core.pipeline_manager.PathManager'):
                # Mock data loader instance
                mock_data_loader = Mock()
                mock_data_loader_class.return_value = mock_data_loader
                
                # Mock data loading methods
                mock_data_loader.load_data.return_value = (sample_adata, sample_adata.copy(), sample_adata.copy())
                mock_data_loader.load_corrected_data.return_value = (sample_adata.copy(), sample_adata.copy())
                mock_data_loader.load_gene_lists.return_value = (['gene1', 'gene2'], ['sexgene1'])
                
                pipeline = PipelineManager(mock_config)
                pipeline.load_data()
                
                # Should load basic and corrected data
                mock_data_loader.load_data.assert_called_once()
                mock_data_loader.load_corrected_data.assert_called_once()
                mock_data_loader.load_gene_lists.assert_called_once()
                
                # Check attributes
                assert hasattr(pipeline, 'adata_corrected')
                assert hasattr(pipeline, 'adata_eval_corrected')
                assert pipeline.adata_corrected is not None
                assert pipeline.adata_eval_corrected is not None
    
    def test_setup_gene_filtering(self, mock_config, sample_adata):
        """Test gene filtering setup."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.GeneFilter') as mock_gene_filter_class:
                    # Mock gene filter instance
                    mock_gene_filter = Mock()
                    mock_gene_filter_class.return_value = mock_gene_filter
                    mock_gene_filter.apply_filter.return_value = (sample_adata, sample_adata.copy(), sample_adata.copy())
                    
                    pipeline = PipelineManager(mock_config)
                    
                    # Set up required attributes
                    pipeline.adata = sample_adata
                    pipeline.adata_eval = sample_adata.copy()
                    pipeline.adata_original = sample_adata.copy()
                    pipeline.autosomal_genes = ['gene1', 'gene2']
                    pipeline.sex_genes = ['sexgene1']
                    
                    pipeline.setup_gene_filtering()
                    
                    # Should create and use gene filter
                    mock_gene_filter_class.assert_called_once()
                    mock_gene_filter.apply_filter.assert_called_once()
                    
                    # Check that data is updated
                    assert hasattr(pipeline, 'gene_filter')
    
    def test_preprocess_data_for_training(self, mock_config, sample_adata):
        """Test data preprocessing for training mode."""
        mock_config.data_processing.model_management.load_model = False
        
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.DataPreprocessor') as mock_preprocessor_class:
                    # Mock preprocessor instance
                    mock_preprocessor = Mock()
                    mock_preprocessor_class.return_value = mock_preprocessor
                    
                    # Mock preprocessing return values
                    train_data = np.random.rand(80, 50)
                    test_data = np.random.rand(20, 50)
                    train_labels = np.random.rand(80, 4)
                    test_labels = np.random.rand(20, 4)
                    
                    mock_preprocessor.prepare_data.return_value = (
                        train_data, test_data, train_labels, test_labels,
                        Mock(), np.random.rand(10, 50), Mock(), True, [], False
                    )
                    
                    pipeline = PipelineManager(mock_config)
                    
                    # Set up required attributes
                    pipeline.adata = sample_adata
                    pipeline.adata_eval = sample_adata.copy()
                    pipeline.adata_original = sample_adata.copy()
                    pipeline.adata_corrected = None
                    pipeline.adata_eval_corrected = None
                    
                    pipeline.preprocess_data()
                    
                    # Should create preprocessor and process data
                    mock_preprocessor_class.assert_called_once_with(mock_config, sample_adata, None)
                    mock_preprocessor.prepare_data.assert_called_once()
                    
                    # Check attributes
                    assert hasattr(pipeline, 'train_data')
                    assert hasattr(pipeline, 'test_data')
                    assert hasattr(pipeline, 'train_labels')
                    assert hasattr(pipeline, 'test_labels')
    
    def test_preprocess_final_eval_data(self, mock_config, sample_adata):
        """Test final evaluation data preprocessing."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.DataPreprocessor') as mock_preprocessor_class:
                    # Mock preprocessor instance
                    mock_preprocessor = Mock()
                    mock_preprocessor_class.return_value = mock_preprocessor
                    
                    # Mock preprocessing return values
                    test_data = np.random.rand(20, 50)
                    test_labels = np.random.rand(20, 4)
                    
                    mock_preprocessor.prepare_final_eval_data.return_value = (
                        test_data, test_labels, Mock()
                    )
                    
                    pipeline = PipelineManager(mock_config)
                    
                    # Set up required attributes
                    pipeline.adata = sample_adata
                    pipeline.adata_corrected = None
                    pipeline.adata_eval = sample_adata.copy()
                    pipeline.adata_eval_corrected = None
                    pipeline.label_encoder = Mock()
                    pipeline.num_features = 50
                    pipeline.scaler = Mock()
                    pipeline.is_scaler_fit = True
                    pipeline.highly_variable_genes = []
                    pipeline.mix_included = False
                    
                    pipeline.preprocess_final_eval_data()
                    
                    # Should create preprocessor and process final eval data
                    mock_preprocessor_class.assert_called_once()
                    mock_preprocessor.prepare_final_eval_data.assert_called_once()
                    
                    # Check attributes
                    assert hasattr(pipeline, 'test_data')
                    assert hasattr(pipeline, 'test_labels')
    
    def test_run_preprocessing_training_mode(self, mock_config, sample_adata):
        """Test run_preprocessing in training mode."""
        mock_config.data_processing.model_management.load_model = False
        
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch.object(PipelineManager, 'setup_gene_filtering') as mock_gene_filtering:
                    with patch.object(PipelineManager, 'preprocess_data') as mock_preprocess:
                        pipeline = PipelineManager(mock_config)
                        
                        pipeline.run_preprocessing()
                        
                        # Should call gene filtering and regular preprocessing
                        mock_gene_filtering.assert_called_once()
                        mock_preprocess.assert_called_once()
    
    def test_run_preprocessing_eval_mode(self, mock_config, sample_adata):
        """Test run_preprocessing in evaluation mode."""
        mock_config.data_processing.model_management.load_model = True
        
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch.object(PipelineManager, 'setup_gene_filtering') as mock_gene_filtering:
                    with patch.object(PipelineManager, 'load_model_components') as mock_load_components:
                        with patch.object(PipelineManager, 'preprocess_final_eval_data') as mock_final_preprocess:
                            pipeline = PipelineManager(mock_config)
                            
                            pipeline.run_preprocessing()
                            
                            # Should call gene filtering, load components, and final preprocessing
                            mock_gene_filtering.assert_called_once()
                            mock_load_components.assert_called_once()
                            mock_final_preprocess.assert_called_once()
    
    def test_load_model_components(self, mock_config):
        """Test loading model components."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.ModelLoader') as mock_model_loader_class:
                    # Mock model loader instance
                    mock_model_loader = Mock()
                    mock_model_loader_class.return_value = mock_model_loader
                    
                    # Mock component loading
                    mock_components = (
                        Mock(), Mock(), True, [], 50, Mock(), False, np.random.rand(10, 50)
                    )
                    mock_model_loader.load_model_components.return_value = mock_components
                    
                    pipeline = PipelineManager(mock_config)
                    pipeline.load_model_components()
                    
                    # Should create model loader and load components
                    mock_model_loader_class.assert_called_once_with(mock_config)
                    mock_model_loader.load_model_components.assert_called_once()
                    
                    # Check attributes are set
                    assert hasattr(pipeline, 'model_loader')
                    assert hasattr(pipeline, 'label_encoder')
                    assert hasattr(pipeline, 'scaler')
    
    def test_load_model(self, mock_config):
        """Test model loading."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.ModelLoader') as mock_model_loader_class:
                    # Mock model loader instance
                    mock_model_loader = Mock()
                    mock_model_loader_class.return_value = mock_model_loader
                    mock_model_loader.load_model.return_value = Mock()
                    
                    pipeline = PipelineManager(mock_config)
                    pipeline.load_model()
                    
                    # Should create model loader and load model
                    mock_model_loader_class.assert_called_once_with(mock_config)
                    mock_model_loader.load_model.assert_called_once()
                    
                    # Check model is set
                    assert hasattr(pipeline, 'model')
    
    def test_build_and_train_model(self, mock_config):
        """Test model building and training."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.ModelBuilder') as mock_model_builder_class:
                    # Mock model builder instance
                    mock_model_builder = Mock()
                    mock_model_builder_class.return_value = mock_model_builder
                    mock_model_builder.run.return_value = (Mock(), Mock())
                    
                    pipeline = PipelineManager(mock_config)
                    
                    # Set up required attributes
                    pipeline.train_data = np.random.rand(80, 50)
                    pipeline.train_labels = np.random.rand(80, 4)
                    pipeline.label_encoder = Mock()
                    pipeline.reference_data = np.random.rand(10, 50)
                    pipeline.scaler = Mock()
                    pipeline.is_scaler_fit = True
                    pipeline.highly_variable_genes = []
                    pipeline.mix_included = False
                    
                    pipeline.build_and_train_model()
                    
                    # Should create model builder and train
                    mock_model_builder_class.assert_called_once()
                    mock_model_builder.run.assert_called_once()
                    
                    # Check attributes are set
                    assert hasattr(pipeline, 'model_builder')
                    assert hasattr(pipeline, 'model')
                    assert hasattr(pipeline, 'history')
    
    def test_load_or_train_model_training_mode(self, mock_config):
        """Test load_or_train_model in training mode."""
        mock_config.data_processing.model_management.load_model = False
        
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch.object(PipelineManager, 'build_and_train_model') as mock_build_train:
                    pipeline = PipelineManager(mock_config)
                    
                    pipeline.load_or_train_model()
                    
                    # Should build and train model
                    mock_build_train.assert_called_once()
    
    def test_load_or_train_model_loading_mode(self, mock_config):
        """Test load_or_train_model in loading mode."""
        mock_config.data_processing.model_management.load_model = True
        
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch.object(PipelineManager, 'load_model') as mock_load:
                    pipeline = PipelineManager(mock_config)
                    
                    pipeline.load_or_train_model()
                    
                    # Should load model
                    mock_load.assert_called_once()
    
    def test_run_metrics(self, mock_config):
        """Test metrics computation."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.Metrics') as mock_metrics_class:
                    # Mock metrics instance
                    mock_metrics = Mock()
                    mock_metrics_class.return_value = mock_metrics
                    
                    pipeline = PipelineManager(mock_config)
                    
                    # Set up required attributes
                    pipeline.model = Mock()
                    pipeline.test_data = np.random.rand(20, 50)
                    pipeline.test_labels = np.random.rand(20, 4)
                    pipeline.label_encoder = Mock()
                    pipeline.path_manager = Mock()
                    
                    pipeline.run_metrics()
                    
                    # Should create metrics and compute
                    mock_metrics_class.assert_called_once()
                    mock_metrics.compute_metrics.assert_called_once()
    
    def test_display_duration_seconds(self, mock_config):
        """Test duration display for seconds."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Test with seconds
                start_time = 0
                end_time = 45
                
                # Should not raise any errors
                pipeline.display_duration(start_time, end_time)
    
    def test_display_duration_minutes(self, mock_config):
        """Test duration display for minutes."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Test with minutes
                start_time = 0
                end_time = 125  # 2 minutes 5 seconds
                
                # Should not raise any errors
                pipeline.display_duration(start_time, end_time)
    
    def test_run_integration_basic(self, mock_config):
        """Test basic run integration (training mode)."""
        mock_config.data_processing.model_management.load_model = False
        mock_config.feature_importance.run_interpreter = False
        mock_config.feature_importance.run_visualization = False
        
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager') as mock_path_manager_class:
                # Mock path manager
                mock_path_manager = Mock()
                mock_path_manager_class.return_value = mock_path_manager
                mock_path_manager.construct_model_directory.return_value = "outputs/models"
                mock_path_manager.get_visualization_directory.return_value = "outputs/results"
                
                with patch.object(PipelineManager, 'setup_gpu'):
                    with patch.object(PipelineManager, 'load_data'):
                        with patch.object(PipelineManager, 'setup_gene_filtering'):
                            with patch.object(PipelineManager, 'preprocess_data'):
                                with patch.object(PipelineManager, 'load_or_train_model'):
                                    with patch.object(PipelineManager, 'run_metrics'):
                                        pipeline = PipelineManager(mock_config)
                                        
                                        result = pipeline.run()
                                        
                                        # Should return result dict
                                        assert isinstance(result, dict)
                                        assert 'model_path' in result
                                        assert 'results_path' in result
                                        assert 'duration' in result
    
    def test_config_attribute_access(self, mock_config):
        """Test configuration attribute access patterns."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Test that config attributes are accessible
                assert hasattr(pipeline.config_instance, 'data')
                assert hasattr(pipeline.config_instance.data, 'tissue')
                assert hasattr(pipeline.config_instance.data, 'batch_correction')
                
                # Test specific config values
                assert pipeline.config_instance.data.tissue == 'head'
                assert pipeline.config_instance.data.encoding_variable == 'age'
    
    def test_error_handling_in_methods(self, mock_config):
        """Test error handling in pipeline methods."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                with patch('shared.core.pipeline_manager.GPUHandler') as mock_gpu_handler:
                    # Mock GPU handler to raise exception
                    mock_gpu_handler.configure.side_effect = Exception("GPU setup failed")
                    
                    pipeline = PipelineManager(mock_config)
                    
                    # Should propagate exceptions
                    with pytest.raises(Exception, match="GPU setup failed"):
                        pipeline.setup_gpu()
    
    def test_batch_correction_conditional_logic(self, mock_config):
        """Test batch correction conditional logic."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Test batch correction enabled check
                mock_config.data.batch_correction.enabled = True
                batch_enabled = getattr(pipeline.config_instance.data.batch_correction, 'enabled', False)
                assert batch_enabled == True
                
                # Test batch correction disabled check
                mock_config.data.batch_correction.enabled = False
                batch_enabled = getattr(pipeline.config_instance.data.batch_correction, 'enabled', False)
                assert batch_enabled == False
    
    def test_model_management_conditional_logic(self, mock_config):
        """Test model management conditional logic."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Test load model enabled check
                mock_config.data_processing.model_management.load_model = True
                load_model = getattr(getattr(pipeline.config_instance.data_processing, 'model_management', None), 'load_model', False)
                assert load_model == True
                
                # Test load model disabled check
                mock_config.data_processing.model_management.load_model = False
                load_model = getattr(getattr(pipeline.config_instance.data_processing, 'model_management', None), 'load_model', False)
                assert load_model == False
    
    def test_feature_importance_conditional_logic(self, mock_config):
        """Test feature importance conditional logic."""
        with patch('shared.core.pipeline_manager.DataLoader'):
            with patch('shared.core.pipeline_manager.PathManager'):
                pipeline = PipelineManager(mock_config)
                
                # Test interpreter enabled check
                mock_config.feature_importance.run_interpreter = True
                run_interpreter = getattr(pipeline.config_instance.feature_importance, 'run_interpreter', True)
                assert run_interpreter == True
                
                # Test visualization enabled check
                mock_config.feature_importance.run_visualization = True
                run_visualization = getattr(pipeline.config_instance.feature_importance, 'run_visualization', True)
                assert run_visualization == True