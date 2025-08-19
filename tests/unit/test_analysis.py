"""Tests for analysis modules (EDA and visualization functionality)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from anndata import AnnData

from src.timeflies.analysis.eda import EDAHandler
from src.timeflies.analysis.visuals import Visualizer


class TestEDAHandler:
    """Test EDAHandler functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.mock_path_manager = Mock()
        self.adata, self.adata_eval, self.adata_original = self.create_sample_data()
        self.adata_corrected, self.adata_eval_corrected = self.create_corrected_data()
        
    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        
        # Data configuration
        config.data = Mock()
        config.data.encoding_variable = 'age'
        
        # Batch correction configuration
        config.data.batch_correction = Mock()
        config.data.batch_correction.enabled = False
        
        return config
        
    def create_sample_data(self):
        """Create sample AnnData objects for testing."""
        n_obs, n_vars = 1000, 2000
        
        # Create expression data
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        # Create observations (cells)
        obs = pd.DataFrame({
            'age': np.random.choice([1, 5, 10, 20], n_obs),
            'sex': np.random.choice(['male', 'female'], n_obs),
            'tissue': np.random.choice(['head', 'body'], n_obs),
            'afca_annotation_broad': np.random.choice([
                'CNS neuron', 'muscle cell', 'epithelial cell'
            ], n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        # Create variables (genes)
        var = pd.DataFrame({
            'gene_type': np.random.choice(['protein_coding', 'lncRNA'], n_vars),
            'highly_variable': np.random.choice([True, False], n_vars, p=[0.2, 0.8])
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata_eval = adata[::2].copy()  # Every other cell for eval
        adata_original = adata.copy()
        
        return adata, adata_eval, adata_original
        
    def create_corrected_data(self):
        """Create batch-corrected sample data."""
        adata_corrected = self.adata.copy()
        adata_eval_corrected = self.adata_eval.copy()
        
        # Add some variation to simulate batch correction
        adata_corrected.X = adata_corrected.X + np.random.normal(0, 0.1, adata_corrected.X.shape)
        adata_eval_corrected.X = adata_eval_corrected.X + np.random.normal(0, 0.1, adata_eval_corrected.X.shape)
        
        return adata_corrected, adata_eval_corrected
        
    @patch('src.timeflies.analysis.eda.VisualizationTools')
    def test_eda_handler_initialization(self, mock_vis_tools):
        """Test EDAHandler initialization."""
        handler = EDAHandler(
            self.mock_config, self.mock_path_manager,
            self.adata, self.adata_eval, self.adata_original,
            self.adata_corrected, self.adata_eval_corrected
        )
        
        assert handler.config == self.mock_config
        assert handler.path_manager == self.mock_path_manager
        assert handler.adata is self.adata
        assert handler.adata_eval is self.adata_eval
        assert handler.adata_original is self.adata_original
        assert handler.adata_corrected is self.adata_corrected
        assert handler.adata_eval_corrected is self.adata_eval_corrected
        
        # Check that VisualizationTools is initialized
        mock_vis_tools.assert_called_once_with(
            config=self.mock_config, path_manager=self.mock_path_manager
        )
        
    @patch('src.timeflies.analysis.eda.VisualizationTools')
    def test_run_eda_uncorrected_data(self, mock_vis_tools):
        """Test running EDA on uncorrected data."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        handler = EDAHandler(
            self.mock_config, self.mock_path_manager,
            self.adata, self.adata_eval, self.adata_original,
            self.adata_corrected, self.adata_eval_corrected
        )
        
        with patch.object(handler, 'eda') as mock_eda:
            handler.run_eda()
            
            # Should call eda 3 times for uncorrected data (train, eval, original)
            assert mock_eda.call_count == 3
            
            # Check the calls
            calls = mock_eda.call_args_list
            
            # First call: training data
            assert calls[0][1]['dataset_name'] == 'train'
            assert calls[0][1]['folder_name'] == 'Training Data'
            assert calls[0][1]['encoding_column'] == 'age'
            
            # Second call: evaluation data
            assert calls[1][1]['dataset_name'] == 'evaluation'
            assert calls[1][1]['folder_name'] == 'Evaluation Data'
            
            # Third call: original data
            assert calls[2][1]['dataset_name'] == 'original'
            assert calls[2][1]['folder_name'] == 'Original Data'
            
    @patch('src.timeflies.analysis.eda.VisualizationTools')
    def test_run_eda_batch_corrected_data(self, mock_vis_tools):
        """Test running EDA on batch-corrected data."""
        # Enable batch correction
        self.mock_config.data.batch_correction.enabled = True
        
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        handler = EDAHandler(
            self.mock_config, self.mock_path_manager,
            self.adata, self.adata_eval, self.adata_original,
            self.adata_corrected, self.adata_eval_corrected
        )
        
        with patch.object(handler, 'eda') as mock_eda:
            handler.run_eda()
            
            # Should call eda 2 times for batch-corrected data (train, eval)
            assert mock_eda.call_count == 2
            
            # Check the calls
            calls = mock_eda.call_args_list
            
            # First call: batch training data
            assert calls[0][1]['dataset_name'] == 'batch_train'
            assert calls[0][1]['folder_name'] == 'Batch Training Data'
            
            # Second call: batch evaluation data
            assert calls[1][1]['dataset_name'] == 'batch_evaluation'
            assert calls[1][1]['folder_name'] == 'Batch Evaluation Data'


class TestVisualizer:
    """Test Visualizer functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.mock_model = Mock()
        self.mock_history = Mock()
        self.test_inputs = np.random.randn(100, 50)
        self.test_labels = np.eye(3)[np.random.choice(3, 100)]
        self.label_encoder = self.create_label_encoder()
        self.squeezed_shap_values = np.random.randn(100, 50, 3)
        self.squeezed_test_data = np.random.randn(100, 50)
        self.adata, self.adata_corrected = self.create_sample_data()
        self.mock_path_manager = Mock()
        
    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        
        # Data configuration
        config.data = Mock()
        config.data.model_type = 'CNN'
        config.data.encoding_variable = 'age'
        
        # Batch correction configuration
        config.data.batch_correction = Mock()
        config.data.batch_correction.enabled = False
        
        # Gene preprocessing configuration
        config.gene_preprocessing = Mock()
        config.gene_preprocessing.gene_filtering = Mock()
        config.gene_preprocessing.gene_filtering.select_batch_genes = False
        
        return config
        
    def create_label_encoder(self):
        """Create a fitted label encoder."""
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(['1', '5', '10'])  # Age classes as strings
        return encoder
        
    def create_sample_data(self):
        """Create sample AnnData objects."""
        n_obs, n_vars = 1000, 2000
        
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        obs = pd.DataFrame({
            'age': np.random.choice([1, 5, 10], n_obs),
            'sex': np.random.choice(['male', 'female'], n_obs),
            'tissue': np.random.choice(['head', 'body'], n_obs),
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        var = pd.DataFrame({
            'gene_type': np.random.choice(['protein_coding', 'lncRNA'], n_vars),
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata_corrected = adata.copy()
        
        return adata, adata_corrected
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualizer_initialization(self, mock_vis_tools):
        """Test Visualizer initialization."""
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        assert visualizer.config == self.mock_config
        assert visualizer.model == self.mock_model
        assert visualizer.history == self.mock_history
        assert np.array_equal(visualizer.test_inputs, self.test_inputs)
        assert np.array_equal(visualizer.test_labels, self.test_labels)
        assert visualizer.label_encoder == self.label_encoder
        assert np.array_equal(visualizer.squeezed_shap_values, self.squeezed_shap_values)
        assert np.array_equal(visualizer.squeezed_test_data, self.squeezed_test_data)
        assert visualizer.adata is self.adata
        assert visualizer.adata_corrected is self.adata_corrected
        assert visualizer.path_manager == self.mock_path_manager
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_import_metrics_neural_network(self, mock_vis_tools):
        """Test importing metrics for neural network models."""
        mock_predictions = np.random.rand(100, 3)
        self.mock_model.predict.return_value = mock_predictions
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        visualizer.import_metrics()
        
        self.mock_model.predict.assert_called_once_with(self.test_inputs)
        assert hasattr(visualizer, 'y_pred')
        assert hasattr(visualizer, 'y_pred_class')
        assert hasattr(visualizer, 'y_true_class')
        assert np.array_equal(visualizer.y_pred, mock_predictions)
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_import_metrics_sklearn_model(self, mock_vis_tools):
        """Test importing metrics for sklearn models."""
        self.mock_config.data.model_type = 'LogisticRegression'
        mock_predictions = np.random.rand(100, 3)
        self.mock_model.predict_proba.return_value = mock_predictions
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        visualizer.import_metrics()
        
        self.mock_model.predict_proba.assert_called_once_with(self.test_inputs)
        assert hasattr(visualizer, 'y_pred')
        assert np.array_equal(visualizer.y_pred, mock_predictions)
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualize_training_history_neural_network(self, mock_vis_tools):
        """Test visualizing training history for neural networks."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        visualizer._visualize_training_history()
        
        mock_visual_tools.plot_history.assert_called_once_with(
            self.mock_history, "training_metrics.png"
        )
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualize_training_history_xgboost(self, mock_vis_tools):
        """Test visualizing training history for XGBoost."""
        self.mock_config.data.model_type = 'XGBoost'
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        visualizer._visualize_training_history()
        
        mock_visual_tools.plot_xgboost_history.assert_called_once_with(
            self.mock_history, "training_metrics.png"
        )
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualize_confusion_matrix(self, mock_vis_tools):
        """Test visualizing confusion matrix."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        # Set up required attributes
        visualizer.y_pred_class = np.random.choice(3, 100)
        visualizer.y_true_class = np.random.choice(3, 100)
        
        visualizer._visualize_confusion_matrix()
        
        mock_visual_tools.create_confusion_matrix.assert_called_once()
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_sort_labels_by_age(self, mock_vis_tools):
        """Test sorting labels by age."""
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        class_labels = ['10', '1', '5', '20']
        sorted_labels = visualizer._sort_labels_by_age(class_labels)
        
        # Should be sorted in ascending age order
        expected_order = ['1', '5', '10', '20']
        assert sorted_labels == expected_order
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualize_shap_summary_with_batch_correction(self, mock_vis_tools):
        """Test SHAP summary visualization with batch correction enabled."""
        self.mock_config.data.batch_correction.enabled = True
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        visualizer._visualize_shap_summary()
        
        # Should use corrected data variable names when batch correction is enabled
        mock_visual_tools.plot_shap_summary.assert_called_once()
        call_args = mock_visual_tools.plot_shap_summary.call_args[1]
        
        # Check that the feature names come from corrected data
        assert 'feature_names' in call_args
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualize_shap_summary_without_batch_correction(self, mock_vis_tools):
        """Test SHAP summary visualization without batch correction."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        visualizer._visualize_shap_summary()
        
        # Should use regular data variable names when batch correction is disabled
        mock_visual_tools.plot_shap_summary.assert_called_once()
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_visualize_shap_summary_no_shap_values(self, mock_vis_tools):
        """Test SHAP summary visualization when SHAP values are None."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            None, None,  # No SHAP values
            self.adata, self.adata_corrected, self.mock_path_manager
        )
        
        visualizer._visualize_shap_summary()
        
        # Should not call plot_shap_summary when SHAP values are None
        mock_visual_tools.plot_shap_summary.assert_not_called()
        
    @patch('src.timeflies.analysis.visuals.VisualizationTools')
    def test_run_visualization_pipeline(self, mock_vis_tools):
        """Test running the complete visualization pipeline."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools
        
        visualizer = Visualizer(
            self.mock_config, self.mock_model, self.mock_history,
            self.test_inputs, self.test_labels, self.label_encoder,
            self.squeezed_shap_values, self.squeezed_test_data,
            self.adata, self.adata_corrected, self.mock_path_manager,
            preserved_var_names=None
        )
        
        # Mock the model predictions
        self.mock_model.predict.return_value = np.random.rand(100, 3)
        
        with patch.object(visualizer, '_visualize_training_history') as mock_history, \
             patch.object(visualizer, '_visualize_confusion_matrix') as mock_confusion, \
             patch.object(visualizer, '_visualize_shap_summary') as mock_shap:
            
            visualizer.run()
            
            # Check that all visualization methods are called
            mock_history.assert_called_once()
            mock_confusion.assert_called_once()
            mock_shap.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])