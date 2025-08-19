"""Tests for model interpretation and metrics functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from sklearn.preprocessing import LabelEncoder
import json
import pickle

from src.timeflies.evaluation.interpreter import Interpreter, Metrics, Prediction


class TestPrediction:
    """Test Prediction static methods."""
    
    def test_calculate_baseline_scores(self):
        """Test baseline score calculation."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        
        baseline_accuracy, baseline_precision, baseline_recall, baseline_f1 = \
            Prediction.calculate_baseline_scores(y_true)
        
        # Baseline should predict most frequent class (0 appears 4 times)
        assert isinstance(baseline_accuracy, float)
        assert isinstance(baseline_precision, float)
        assert isinstance(baseline_recall, float)
        assert isinstance(baseline_f1, float)
        assert 0.0 <= baseline_accuracy <= 1.0
        
    def test_evaluate_model(self):
        """Test model evaluation."""
        mock_model = Mock()
        mock_model.evaluate.return_value = (0.5, 0.85, 0.9)  # loss, accuracy, auc
        
        test_inputs = np.random.randn(100, 50)
        test_labels = np.random.randint(0, 3, (100, 3))
        
        result = Prediction.evaluate_model(
            mock_model, test_inputs, test_labels
        )
        
        # Should return a tuple from model.evaluate unpacking
        assert result == (0.5, 0.85, 0.9)
        mock_model.evaluate.assert_called_once_with(test_inputs, test_labels)
        
    def test_make_predictions(self):
        """Test making predictions."""
        mock_model = Mock()
        test_inputs = np.random.randn(100, 50)
        
        # Mock prediction output
        mock_predictions = np.random.rand(100, 3)  # 3 classes
        mock_model.predict.return_value = mock_predictions
        
        y_pred, y_pred_binary = Prediction.make_predictions(mock_model, test_inputs)
        
        assert np.array_equal(y_pred, mock_predictions)
        assert y_pred_binary.shape == (100,)
        assert all(0 <= pred <= 2 for pred in y_pred_binary)  # Should be class indices
        mock_model.predict.assert_called_once_with(test_inputs)


class TestInterpreter:
    """Test Interpreter functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.mock_model = Mock()
        self.test_data = np.random.randn(100, 50)
        self.test_labels = np.eye(3)[np.random.choice(3, 100)]
        self.label_encoder = self.create_label_encoder()
        self.reference_data = np.random.randn(50, 50)
        self.mock_path_manager = Mock()
        self.mock_path_manager.get_visualization_directory.return_value = '/tmp/shap'
        
    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        
        # Feature importance configuration
        config.feature_importance = Mock()
        config.feature_importance.load_SHAP = False
        
        # Data configuration
        config.data = Mock()
        config.data.model_type = 'CNN'
        
        # Device configuration
        config.device = Mock()
        config.device.processor = 'GPU'
        
        return config
        
    def create_label_encoder(self):
        """Create a fitted label encoder."""
        encoder = LabelEncoder()
        encoder.fit(['class_0', 'class_1', 'class_2'])
        return encoder
        
    def test_interpreter_initialization(self):
        """Test Interpreter initialization."""
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        assert interpreter.config == self.mock_config
        assert interpreter.model == self.mock_model
        assert np.array_equal(interpreter.test_data, self.test_data)
        assert np.array_equal(interpreter.test_labels, self.test_labels)
        assert interpreter.label_encoder == self.label_encoder
        assert np.array_equal(interpreter.reference_data, self.reference_data)
        assert interpreter.path_manager == self.mock_path_manager
        assert interpreter.shap_dir == '/tmp/shap'
        
    @patch('shap.GradientExplainer')
    def test_compute_shap_values_neural_network(self, mock_gradient_explainer):
        """Test SHAP value computation for neural networks."""
        mock_explainer = Mock()
        mock_gradient_explainer.return_value = mock_explainer
        mock_shap_values = np.random.randn(100, 50, 3)
        mock_explainer.shap_values.return_value = mock_shap_values
        
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        shap_values, squeezed_test_data = interpreter.compute_shap_values()
        
        mock_gradient_explainer.assert_called_once_with(self.mock_model, self.reference_data)
        mock_explainer.shap_values.assert_called_once_with(self.test_data)
        assert isinstance(shap_values, list)  # Should be converted to list for GPU
        assert np.array_equal(squeezed_test_data, self.test_data)
        
    @patch('shap.TreeExplainer')
    def test_compute_shap_values_tree_model(self, mock_tree_explainer):
        """Test SHAP value computation for tree models."""
        self.mock_config.data.model_type = 'XGBoost'
        mock_explainer = Mock()
        mock_tree_explainer.return_value = mock_explainer
        mock_shap_values = np.random.randn(100, 50, 3)
        mock_explainer.shap_values.return_value = mock_shap_values
        
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        shap_values, squeezed_test_data = interpreter.compute_shap_values()
        
        mock_tree_explainer.assert_called_once_with(self.mock_model)
        mock_explainer.shap_values.assert_called_once_with(self.test_data)
        
    @patch('shap.LinearExplainer')
    def test_compute_shap_values_linear_model(self, mock_linear_explainer):
        """Test SHAP value computation for linear models."""
        self.mock_config.data.model_type = 'LogisticRegression'
        mock_explainer = Mock()
        mock_linear_explainer.return_value = mock_explainer
        mock_shap_values = np.random.randn(100, 50, 3)  # Need 3D for linear model
        mock_explainer.shap_values.return_value = mock_shap_values
        
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        result = interpreter.compute_shap_values()
        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        mock_linear_explainer.assert_called_once_with(self.mock_model, self.reference_data)
        mock_explainer.shap_values.assert_called_once_with(self.test_data)
        
    def test_compute_shap_values_mac_processor(self):
        """Test SHAP value computation on Mac processor."""
        self.mock_config.device.processor = 'M'
        
        with patch('shap.GradientExplainer') as mock_explainer_class:
            mock_explainer = Mock()
            mock_explainer_class.return_value = mock_explainer
            mock_shap_values = np.random.randn(100, 1, 50, 3)  # 4D for Mac processing
            mock_explainer.shap_values.return_value = mock_shap_values
            
            interpreter = Interpreter(
                self.mock_config, self.mock_model, self.test_data,
                self.test_labels, self.label_encoder, self.reference_data,
                self.mock_path_manager
            )
            
            shap_values, squeezed_test_data = interpreter.compute_shap_values()
            
            # Should squeeze the extra dimension for Mac
            assert isinstance(shap_values, np.ndarray)
            assert shap_values.ndim == 3  # Should be squeezed from 4D to 3D
            
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    @patch('json.dump')
    def test_save_shap_values(self, mock_json_dump, mock_pickle_dump, mock_file):
        """Test saving SHAP values."""
        # Mock model with weights method
        self.mock_model.get_weights.return_value = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        mock_shap_values = np.random.randn(100, 50, 3)
        interpreter.save_shap_values(mock_shap_values)
        
        # Check that pickle.dump was called
        mock_pickle_dump.assert_called_once()
        
        # Check that the file was opened for writing
        mock_file.assert_called()
        
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_load_shap_values(self, mock_pickle_load, mock_file, mock_exists):
        """Test loading SHAP values."""
        mock_exists.return_value = True
        
        # Mock loaded data
        mock_data = {
            'shap_values': np.random.randn(100, 50, 3),
            'metadata': {
                'model_type': 'CNN',
                'model_weights_hash': 'hash123',
                'test_data_hash': 'testhash',
                'reference_data_hash': 'refhash'
            },
            'reference_data': self.reference_data,
            'test_data': self.test_data
        }
        mock_pickle_load.return_value = mock_data
        
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        with patch.object(interpreter, '_get_model_weights_hash', return_value='hash123'), \
             patch.object(interpreter, 'compute_sha256_hash', side_effect=['testhash', 'refhash']):
            
            loaded_shap_values = interpreter.load_shap_values()
            
            assert np.array_equal(loaded_shap_values, mock_data['shap_values'])
            mock_pickle_load.assert_called_once()
            
    def test_compute_sha256_hash(self):
        """Test SHA-256 hash computation."""
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        test_data = b"test data"
        hash_result = interpreter.compute_sha256_hash(test_data)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 produces 64-character hex string
        
    @patch.object(Interpreter, 'compute_shap_values')
    @patch.object(Interpreter, 'save_shap_values')
    def test_compute_or_load_shap_values_compute(self, mock_save, mock_compute):
        """Test computing SHAP values when not loading from disk."""
        mock_shap_values = np.random.randn(100, 50, 3)
        mock_test_data = self.test_data
        mock_compute.return_value = (mock_shap_values, mock_test_data)
        
        interpreter = Interpreter(
            self.mock_config, self.mock_model, self.test_data,
            self.test_labels, self.label_encoder, self.reference_data,
            self.mock_path_manager
        )
        
        shap_values, test_data = interpreter.compute_or_load_shap_values()
        
        mock_compute.assert_called_once()
        mock_save.assert_called_once_with(mock_shap_values)
        assert np.array_equal(shap_values, mock_shap_values)
        assert np.array_equal(test_data, mock_test_data)


class TestMetrics:
    """Test Metrics functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.mock_model = Mock()
        self.test_inputs = np.random.randn(100, 50)
        self.test_labels = np.eye(3)[np.random.choice(3, 100)]
        self.label_encoder = self.create_label_encoder()
        self.mock_path_manager = Mock()
        self.mock_path_manager.get_visualization_directory.return_value = '/tmp/results'
        
    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        
        # Data configuration
        config.data = Mock()
        config.data.model_type = 'CNN'
        
        # Feature importance configuration
        config.feature_importance = Mock()
        config.feature_importance.run_interpreter = True
        config.feature_importance.save_predictions = True
        
        # Train-test split configuration
        config.data.train_test_split = Mock()
        config.data.train_test_split.method = 'random'
        config.data.train_test_split.train = Mock()
        config.data.train_test_split.train.random = 'male'
        config.data.train_test_split.test = Mock()
        config.data.train_test_split.test.random = 'female'
        
        return config
        
    def create_label_encoder(self):
        """Create a fitted label encoder."""
        encoder = LabelEncoder()
        encoder.fit(['class_0', 'class_1', 'class_2'])
        return encoder
        
    def test_metrics_initialization(self):
        """Test Metrics initialization."""
        metrics = Metrics(
            self.mock_config, self.mock_model, self.test_inputs,
            self.test_labels, self.label_encoder, self.mock_path_manager
        )
        
        assert metrics.config == self.mock_config
        assert metrics.model == self.mock_model
        assert np.array_equal(metrics.test_inputs, self.test_inputs)
        assert np.array_equal(metrics.test_labels, self.test_labels)
        assert metrics.label_encoder == self.label_encoder
        assert metrics.path_manager == self.mock_path_manager
        assert metrics.output_dir == '/tmp/results'
        
    @patch.object(Prediction, 'evaluate_model')
    def test_evaluate_model_performance_neural_network(self, mock_evaluate):
        """Test model performance evaluation for neural networks."""
        mock_evaluate.return_value = (0.5, 0.85, 0.9)
        
        # Mock model predictions
        mock_predictions = np.random.rand(100, 3)
        self.mock_model.predict.return_value = mock_predictions
        
        metrics = Metrics(
            self.mock_config, self.mock_model, self.test_inputs,
            self.test_labels, self.label_encoder, self.mock_path_manager
        )
        
        metrics._evaluate_model_performance()
        
        mock_evaluate.assert_called_once_with(
            self.mock_model, self.test_inputs, self.test_labels
        )
        self.mock_model.predict.assert_called_once_with(self.test_inputs)
        assert hasattr(metrics, 'y_pred')
        assert hasattr(metrics, 'y_pred_class')
        assert hasattr(metrics, 'y_true_class')
        
    def test_evaluate_model_performance_sklearn(self):
        """Test model performance evaluation for sklearn models."""
        self.mock_config.data.model_type = 'LogisticRegression'
        
        # Mock model predictions
        mock_predictions = np.random.rand(100, 3)
        self.mock_model.predict_proba.return_value = mock_predictions
        
        metrics = Metrics(
            self.mock_config, self.mock_model, self.test_inputs,
            self.test_labels, self.label_encoder, self.mock_path_manager
        )
        
        metrics._evaluate_model_performance()
        
        self.mock_model.predict_proba.assert_called_once_with(self.test_inputs)
        assert hasattr(metrics, 'y_pred')
        assert hasattr(metrics, 'y_pred_class')
        assert hasattr(metrics, 'y_true_class')
        
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('os.makedirs')
    def test_save_metrics_as_json(self, mock_makedirs, mock_json_dump, mock_file):
        """Test saving metrics as JSON."""
        metrics = Metrics(
            self.mock_config, self.mock_model, self.test_inputs,
            self.test_labels, self.label_encoder, self.mock_path_manager
        )
        
        metrics.save_metrics_as_json(
            test_accuracy=0.85,
            test_precision=0.82,
            test_recall=0.88,
            test_f1=0.85,
            test_auc=0.90,
            baseline_accuracy=0.33,
            baseline_precision=0.30,
            baseline_recall=0.33,
            baseline_f1=0.32,
            file_name="test_stats.json"
        )
        
        # Check that directories are created
        mock_makedirs.assert_called()
        
        # Check that JSON is dumped
        mock_json_dump.assert_called()
        
        # Check the structure of saved metrics
        saved_metrics = mock_json_dump.call_args[0][0]
        assert 'Test' in saved_metrics
        assert 'Baseline' in saved_metrics
        assert 'Accuracy' in saved_metrics['Test']
        assert 'Precision' in saved_metrics['Test']
        
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.join')
    def test_save_predictions_to_csv(self, mock_join, mock_to_csv):
        """Test saving predictions to CSV."""
        mock_join.return_value = '/tmp/results/predictions.csv'
        
        metrics = Metrics(
            self.mock_config, self.mock_model, self.test_inputs,
            self.test_labels, self.label_encoder, self.mock_path_manager
        )
        
        # Set up predictions
        metrics.y_pred = np.random.rand(100, 3)
        metrics.y_pred_class = np.random.choice(3, 100)
        metrics.y_true_class = np.random.choice(3, 100)
        
        metrics.save_predictions_to_csv()
        
        mock_to_csv.assert_called_once_with('/tmp/results/predictions.csv', index=False)
        
    @patch.object(Metrics, '_evaluate_model_performance')
    @patch.object(Metrics, '_calculate_and_save_metrics')
    @patch.object(Metrics, 'save_predictions_to_csv')
    def test_compute_metrics(self, mock_save_pred, mock_calc_save, mock_eval):
        """Test complete metrics computation pipeline."""
        metrics = Metrics(
            self.mock_config, self.mock_model, self.test_inputs,
            self.test_labels, self.label_encoder, self.mock_path_manager
        )
        
        metrics.compute_metrics()
        
        mock_eval.assert_called_once()
        mock_calc_save.assert_called_once()
        mock_save_pred.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])