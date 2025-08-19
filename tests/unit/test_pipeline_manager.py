"""Tests for pipeline manager functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from anndata import AnnData

from src.timeflies.core.pipeline_manager import PipelineManager


class TestPipelineManager:
    """Test PipelineManager functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        
    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        
        # Data processing configuration
        config.data_processing = Mock()
        config.data_processing.exploratory_data_analysis = Mock()
        config.data_processing.exploratory_data_analysis.enabled = False
        config.data_processing.model_management = Mock()
        config.data_processing.model_management.load_model = False
        
        # Data configuration
        config.data = Mock()
        config.data.batch_correction = Mock()
        config.data.batch_correction.enabled = False
        
        # Data processing configuration - set up to return False for EDA enabled check
        config.data_processing = Mock()
        eda_config = Mock()
        eda_config.get.return_value = False  # Make .get('enabled', False) return False
        config.data_processing.exploratory_data_analysis = eda_config
        config.data_processing.model_management = Mock()
        config.data_processing.model_management.load_model = False
        
        # Feature importance configuration
        config.feature_importance = Mock()
        config.feature_importance.run_interpreter = True
        config.feature_importance.run_visualization = True
        
        return config
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    def test_pipeline_manager_initialization(self, mock_path_manager, mock_data_loader):
        """Test PipelineManager initialization."""
        manager = PipelineManager(self.mock_config)
        
        assert manager.config_instance == self.mock_config
        mock_data_loader.assert_called_once_with(self.mock_config)
        mock_path_manager.assert_called_once_with(self.mock_config)
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    def test_setup_gpu_success(self, mock_gpu_handler, mock_path_manager, mock_data_loader):
        """Test successful GPU setup."""
        manager = PipelineManager(self.mock_config)
        
        manager.setup_gpu()
        
        mock_gpu_handler.configure.assert_called_once_with(self.mock_config)
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.GPUHandler')
    def test_setup_gpu_failure(self, mock_gpu_handler, mock_path_manager, mock_data_loader):
        """Test GPU setup failure."""
        mock_gpu_handler.configure.side_effect = Exception("GPU setup failed")
        
        manager = PipelineManager(self.mock_config)
        
        with pytest.raises(Exception, match="GPU setup failed"):
            manager.setup_gpu()
            
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    def test_load_data_success(self, mock_path_manager, mock_data_loader):
        """Test successful data loading."""
        # Enable batch correction for this test
        self.mock_config.data.batch_correction.enabled = True
        
        # Mock data loader methods
        mock_loader_instance = mock_data_loader.return_value
        mock_loader_instance.load_data.return_value = (Mock(), Mock(), Mock())
        mock_loader_instance.load_corrected_data.return_value = (Mock(), Mock())
        mock_loader_instance.load_gene_lists.return_value = (['gene1'], ['sex_gene1'])
        
        manager = PipelineManager(self.mock_config)
        manager.load_data()
        
        # Check that all loading methods were called
        mock_loader_instance.load_data.assert_called_once()
        mock_loader_instance.load_corrected_data.assert_called_once()
        mock_loader_instance.load_gene_lists.assert_called_once()
        
        # Check that attributes are set
        assert hasattr(manager, 'adata')
        assert hasattr(manager, 'adata_eval')
        assert hasattr(manager, 'adata_original')
        assert hasattr(manager, 'adata_corrected')
        assert hasattr(manager, 'adata_eval_corrected')
        assert hasattr(manager, 'autosomal_genes')
        assert hasattr(manager, 'sex_genes')
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    def test_load_data_failure(self, mock_path_manager, mock_data_loader):
        """Test data loading failure."""
        mock_loader_instance = mock_data_loader.return_value
        mock_loader_instance.load_data.side_effect = Exception("Data loading failed")
        
        manager = PipelineManager(self.mock_config)
        
        with pytest.raises(Exception, match="Data loading failed"):
            manager.load_data()
            
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.EDAHandler')
    def test_run_eda_enabled(self, mock_eda_handler, mock_path_manager, mock_data_loader):
        """Test running EDA when enabled."""
        # Set up EDA config to return True
        eda_config = Mock()
        eda_config.get.return_value = True  # Make .get('enabled', False) return True
        self.mock_config.data_processing.exploratory_data_analysis = eda_config
        
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.adata = Mock()
        manager.adata_eval = Mock()
        manager.adata_original = Mock()
        manager.adata_corrected = Mock()
        manager.adata_eval_corrected = Mock()
        
        mock_eda_instance = mock_eda_handler.return_value
        
        manager.run_eda()
        
        mock_eda_handler.assert_called_once()
        mock_eda_instance.run_eda.assert_called_once()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.EDAHandler')
    def test_run_eda_disabled(self, mock_eda_handler, mock_path_manager, mock_data_loader):
        """Test skipping EDA when disabled."""
        # EDA is disabled by default in mock config
        manager = PipelineManager(self.mock_config)
        
        manager.run_eda()
        
        # EDAHandler should not be called when disabled
        mock_eda_handler.assert_not_called()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.GeneFilter')
    def test_setup_gene_filtering(self, mock_gene_filter, mock_path_manager, mock_data_loader):
        """Test gene filtering setup."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.adata = Mock()
        manager.adata_eval = Mock()
        manager.adata_original = Mock()
        manager.autosomal_genes = ['gene1']
        manager.sex_genes = ['sex_gene1']
        
        mock_filter_instance = mock_gene_filter.return_value
        mock_filter_instance.apply_filter.return_value = (Mock(), Mock(), Mock())
        
        manager.setup_gene_filtering()
        
        mock_gene_filter.assert_called_once()
        mock_filter_instance.apply_filter.assert_called_once()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.DataPreprocessor')
    def test_preprocess_data(self, mock_data_preprocessor, mock_path_manager, mock_data_loader):
        """Test data preprocessing."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes that preprocess_data expects
        manager.adata = Mock()
        manager.adata_eval = Mock()
        manager.adata_original = Mock()
        manager.adata_corrected = Mock()
        
        mock_preprocessor_instance = mock_data_preprocessor.return_value
        mock_preprocessor_instance.prepare_data.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock()
        )
        
        # Capture references before they get deleted
        adata_ref = manager.adata
        adata_corrected_ref = manager.adata_corrected
        
        manager.preprocess_data()
        
        mock_data_preprocessor.assert_called_once_with(
            self.mock_config, adata_ref, adata_corrected_ref
        )
        mock_preprocessor_instance.prepare_data.assert_called_once()
        
        # Check that preprocessing output attributes are set
        assert hasattr(manager, 'train_data')
        assert hasattr(manager, 'test_data')
        assert hasattr(manager, 'train_labels')
        assert hasattr(manager, 'test_labels')
        assert hasattr(manager, 'label_encoder')
        assert hasattr(manager, 'reference_data')
        
        # Raw data attributes should be deleted for memory cleanup
        assert not hasattr(manager, 'adata')
        assert not hasattr(manager, 'adata_eval')
        assert not hasattr(manager, 'adata_original')
        assert hasattr(manager, 'scaler')
        assert hasattr(manager, 'is_scaler_fit')
        assert hasattr(manager, 'highly_variable_genes')
        assert hasattr(manager, 'mix_included')
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.DataPreprocessor')
    def test_preprocess_final_eval_data(self, mock_data_preprocessor, mock_path_manager, mock_data_loader):
        """Test final evaluation data preprocessing."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.adata = Mock()
        manager.adata_corrected = Mock()
        manager.adata_eval = Mock()
        manager.adata_eval_corrected = Mock()
        manager.label_encoder = Mock()
        manager.num_features = 100
        manager.scaler = Mock()
        manager.is_scaler_fit = True
        manager.highly_variable_genes = ['gene1']
        manager.mix_included = False
        
        mock_preprocessor_instance = mock_data_preprocessor.return_value
        mock_preprocessor_instance.prepare_final_eval_data.return_value = (Mock(), Mock(), Mock())
        
        manager.preprocess_final_eval_data()
        
        mock_preprocessor_instance.prepare_final_eval_data.assert_called_once()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.ModelLoader')
    def test_load_model_components(self, mock_model_loader, mock_path_manager, mock_data_loader):
        """Test loading model components."""
        manager = PipelineManager(self.mock_config)
        
        mock_loader_instance = mock_model_loader.return_value
        mock_loader_instance.load_model_components.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock()
        )
        
        manager.load_model_components()
        
        mock_model_loader.assert_called_once_with(self.mock_config)
        mock_loader_instance.load_model_components.assert_called_once()
        
        # Check that all attributes are set
        assert hasattr(manager, 'label_encoder')
        assert hasattr(manager, 'scaler')
        assert hasattr(manager, 'is_scaler_fit')
        assert hasattr(manager, 'highly_variable_genes')
        assert hasattr(manager, 'num_features')
        assert hasattr(manager, 'history')
        assert hasattr(manager, 'mix_included')
        assert hasattr(manager, 'reference_data')
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.ModelBuilder')
    def test_build_and_train_model(self, mock_model_builder, mock_path_manager, mock_data_loader):
        """Test building and training model."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.train_data = Mock()
        manager.train_labels = Mock()
        manager.label_encoder = Mock()
        manager.reference_data = Mock()
        manager.scaler = Mock()
        manager.is_scaler_fit = True
        manager.highly_variable_genes = ['gene1']
        manager.mix_included = False
        
        mock_builder_instance = mock_model_builder.return_value
        mock_history = Mock()
        mock_history.history = {'loss': [0.5, 0.3], 'accuracy': [0.8, 0.9]}
        mock_builder_instance.run.return_value = (Mock(), mock_history)
        
        manager.build_and_train_model()
        
        mock_model_builder.assert_called_once()
        mock_builder_instance.run.assert_called_once()
        
        assert hasattr(manager, 'model')
        assert hasattr(manager, 'history')
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    def test_load_or_train_model_train(self, mock_path_manager, mock_data_loader):
        """Test loading or training model - training path."""
        manager = PipelineManager(self.mock_config)
        
        with patch.object(manager, 'build_and_train_model') as mock_build_train:
            manager.load_or_train_model()
            mock_build_train.assert_called_once()
            
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    def test_load_or_train_model_load(self, mock_path_manager, mock_data_loader):
        """Test loading or training model - loading path."""
        # Enable model loading in config
        self.mock_config.data_processing.model_management.load_model = True
        
        manager = PipelineManager(self.mock_config)
        
        with patch.object(manager, 'load_model') as mock_load:
            manager.load_or_train_model()
            mock_load.assert_called_once()
            
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.Interpreter')
    def test_run_interpretation_enabled(self, mock_interpreter, mock_path_manager, mock_data_loader):
        """Test running SHAP interpretation when enabled."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.model = Mock()
        manager.test_data = Mock()
        manager.test_labels = Mock()
        manager.label_encoder = Mock()
        manager.reference_data = Mock()
        manager.path_manager = Mock()
        
        mock_interpreter_instance = mock_interpreter.return_value
        mock_interpreter_instance.compute_or_load_shap_values.return_value = (Mock(), Mock())
        
        manager.run_interpretation()
        
        mock_interpreter.assert_called_once()
        mock_interpreter_instance.compute_or_load_shap_values.assert_called_once()
        
        assert hasattr(manager, 'squeezed_shap_values')
        assert hasattr(manager, 'squeezed_test_data')
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.Interpreter')
    def test_run_interpretation_disabled(self, mock_interpreter, mock_path_manager, mock_data_loader):
        """Test skipping SHAP interpretation when disabled."""
        # Disable interpretation in config
        self.mock_config.feature_importance.run_interpreter = False
        
        manager = PipelineManager(self.mock_config)
        
        manager.run_interpretation()
        
        # Interpreter should not be called when disabled
        mock_interpreter.assert_not_called()
        assert manager.squeezed_shap_values is None
        assert manager.squeezed_test_data is None
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.Visualizer')
    def test_run_visualizations_enabled(self, mock_visualizer, mock_path_manager, mock_data_loader):
        """Test running visualizations when enabled."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.model = Mock()
        manager.history = Mock()
        manager.test_data = Mock()
        manager.test_labels = Mock()
        manager.label_encoder = Mock()
        manager.squeezed_shap_values = Mock()
        manager.squeezed_test_data = Mock()
        manager.adata = Mock()
        manager.adata_corrected = Mock()
        manager.path_manager = Mock()
        
        mock_visualizer_instance = mock_visualizer.return_value
        
        manager.run_visualizations()
        
        mock_visualizer.assert_called_once()
        mock_visualizer_instance.run.assert_called_once()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.Visualizer')
    def test_run_visualizations_disabled(self, mock_visualizer, mock_path_manager, mock_data_loader):
        """Test skipping visualizations when disabled."""
        # Disable visualization in config
        self.mock_config.feature_importance.run_visualization = False
        
        manager = PipelineManager(self.mock_config)
        
        manager.run_visualizations()
        
        # Visualizer should not be called when disabled
        mock_visualizer.assert_not_called()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.Metrics')
    def test_run_metrics(self, mock_metrics, mock_path_manager, mock_data_loader):
        """Test running metrics computation."""
        manager = PipelineManager(self.mock_config)
        
        # Set up required attributes
        manager.model = Mock()
        manager.test_data = Mock()
        manager.test_labels = Mock()
        manager.label_encoder = Mock()
        manager.path_manager = Mock()
        
        mock_metrics_instance = mock_metrics.return_value
        
        manager.run_metrics()
        
        mock_metrics.assert_called_once()
        mock_metrics_instance.compute_metrics.assert_called_once()
        
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.logger')
    def test_display_duration_seconds(self, mock_logger, mock_path_manager, mock_data_loader):
        """Test displaying duration in seconds."""
        manager = PipelineManager(self.mock_config)
        
        manager.display_duration(0, 45)  # 45 seconds
        
        # Should log in seconds format
        mock_logger.info.assert_called_with("The task took 45 seconds.")
            
    @patch('src.timeflies.core.pipeline_manager.DataLoader')
    @patch('src.timeflies.core.pipeline_manager.PathManager')
    @patch('src.timeflies.core.pipeline_manager.logger')
    def test_display_duration_minutes(self, mock_logger, mock_path_manager, mock_data_loader):
        """Test displaying duration in minutes and seconds."""
        manager = PipelineManager(self.mock_config)
        
        manager.display_duration(0, 125)  # 2 minutes 5 seconds
        
        # Should log in minutes and seconds format
        mock_logger.info.assert_called_with("The task took 2 minutes and 5 seconds.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])