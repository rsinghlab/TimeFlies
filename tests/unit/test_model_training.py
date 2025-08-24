"""Tests for model building and training functionality."""

from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pytest
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from common.models.model import CustomModelCheckpoint, ModelBuilder, ModelLoader


class TestModelBuilder:
    """Test ModelBuilder functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.train_data, self.train_labels = self.create_sample_data()
        self.label_encoder = self.create_label_encoder()
        self.reference_data = np.random.randn(50, 100)
        self.scaler = Mock()
        self.is_scaler_fit = True
        self.highly_variable_genes = ["gene1", "gene2", "gene3"]
        self.mix_included = False

    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()

        # Data configuration
        config.data = Mock()
        config.data.model = "CNN"

        # Device configuration
        config.device = Mock()
        config.device.processor = "GPU"

        # Model configuration
        config.model = Mock()

        # CNN configuration
        config.model.cnn = Mock()
        config.model.cnn.filters = [64, 128, 256]
        config.model.cnn.kernel_sizes = [3, 3, 3]
        config.model.cnn.strides = [1, 1, 1]
        config.model.cnn.paddings = ["same", "same", "same"]
        config.model.cnn.pool_sizes = [2, 2, None]
        config.model.cnn.pool_strides = [2, 2, 1]
        config.model.cnn.dense_units = [128, 64]
        config.model.cnn.activation_function = "relu"
        config.model.cnn.dropout_rate = 0.3
        config.model.cnn.learning_rate = 0.001

        # MLP configuration
        config.model.mlp = Mock()
        config.model.mlp.units = [128, 64, 32]
        config.model.mlp.activation_function = "relu"
        config.model.mlp.dropout_rate = 0.3
        config.model.mlp.learning_rate = 0.001

        # Logistic Regression configuration
        config.model.logistic_regression = Mock()
        config.model.logistic_regression.penalty = "l2"
        config.model.logistic_regression.solver = "lbfgs"
        config.model.logistic_regression.l1_ratio = None
        config.model.logistic_regression.random_state = 42
        config.model.logistic_regression.max_iter = 1000

        # Random Forest configuration
        config.model.random_forest = Mock()
        config.model.random_forest.n_estimators = 100
        config.model.random_forest.criterion = "gini"
        config.model.random_forest.max_depth = None
        config.model.random_forest.min_samples_split = 2
        config.model.random_forest.min_samples_leaf = 1
        config.model.random_forest.max_features = "sqrt"
        config.model.random_forest.bootstrap = True
        config.model.random_forest.oob_score = False
        config.model.random_forest.n_jobs = -1
        config.model.random_forest.random_state = 42

        # XGBoost configuration
        config.model.xgboost = Mock()
        config.model.xgboost.learning_rate = 0.1
        config.model.xgboost.n_estimators = 100
        config.model.xgboost.max_depth = 6
        config.model.xgboost.min_child_weight = 1
        config.model.xgboost.subsample = 0.8
        config.model.xgboost.colsample_bytree = 0.8
        config.model.xgboost.random_state = 42
        config.model.xgboost.tree_method = "auto"
        config.model.xgboost.predictor = "auto"
        config.model.xgboost.early_stopping_rounds = 10

        # Training configuration
        config.model.training = Mock()
        config.model.training.custom_loss = "sparse_categorical_crossentropy"
        config.model.training.metrics = ["accuracy"]
        config.model.training.early_stopping_patience = 10
        config.model.training.epochs = 100
        config.model.training.batch_size = 32
        config.model.training.validation_split = 0.2

        # Train-test split configuration
        config.data.train_test_split = Mock()
        config.data.train_test_split.random_state = 42

        return config

    def create_sample_data(self):
        """Create sample training data."""
        # CNN data (3D: samples, time_steps, features)
        train_data_cnn = np.random.randn(1000, 1, 100)
        # MLP data (2D: samples, features)
        train_data_mlp = np.random.randn(1000, 100)

        # One-hot encoded labels for 3 classes
        train_labels = np.eye(3)[np.random.choice(3, 1000)]

        return (train_data_cnn, train_data_mlp), train_labels

    def create_label_encoder(self):
        """Create a fitted label encoder."""
        encoder = LabelEncoder()
        encoder.fit(["class_0", "class_1", "class_2"])
        return encoder

    def test_model_builder_initialization(self):
        """Test ModelBuilder initialization."""
        train_data, _ = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        assert builder.config == self.mock_config
        assert np.array_equal(builder.train_data, train_data)
        assert np.array_equal(builder.train_labels, self.train_labels)
        assert builder.label_encoder == self.label_encoder
        assert builder.model_type == "cnn"  # Should be lowercased

    def test_create_cnn_model(self):
        """Test CNN model creation."""
        train_data, _ = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.create_cnn_model(num_output_units=3)

        assert isinstance(model, tf.keras.Sequential)
        assert len(model.layers) > 0
        assert model.layers[-1].units == 3  # Output layer should have 3 units
        assert model.layers[-1].activation.__name__ == "softmax"

    def test_create_mlp_model(self):
        """Test MLP model creation."""
        _, train_data = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )
        # Model type is set from config automatically

        model = builder.create_mlp_model(num_output_units=3)

        assert isinstance(model, tf.keras.Sequential)
        assert len(model.layers) > 0
        assert model.layers[-1].units == 3
        assert model.layers[-1].activation.__name__ == "softmax"

    def test_create_logistic_regression(self):
        """Test Logistic Regression model creation."""
        train_data, _ = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.create_logistic_regression()

        # Check that it's a LogisticRegression instance
        from sklearn.linear_model import LogisticRegression

        assert isinstance(model, LogisticRegression)
        assert model.penalty == "l2"
        assert model.solver == "lbfgs"
        assert model.random_state == 42

    def test_create_random_forest(self):
        """Test Random Forest model creation."""
        train_data, _ = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.create_random_forest()

        # Check that it's a RandomForestClassifier instance
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
        assert model.criterion == "gini"
        assert model.random_state == 42

    def test_create_xgboost_model(self):
        """Test XGBoost model creation."""
        train_data, _ = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.create_xgboost_model()

        # Check that it's an XGBClassifier instance
        import xgboost as xgb

        assert isinstance(model, xgb.XGBClassifier)
        assert model.learning_rate == 0.1
        assert model.n_estimators == 100
        assert model.random_state == 42

    def test_build_model_cnn(self):
        """Test building CNN model through build_model method."""
        train_data, _ = self.train_data
        self.mock_config.data.model = "CNN"
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.build_model()
        assert isinstance(model, tf.keras.Sequential)

    def test_build_model_mlp(self):
        """Test building MLP model through build_model method."""
        _, train_data = self.train_data
        self.mock_config.data.model = "MLP"
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.build_model()
        assert isinstance(model, tf.keras.Sequential)

    def test_build_model_logistic_regression(self):
        """Test building Logistic Regression model through build_model method."""
        train_data, _ = self.train_data
        self.mock_config.data.model = "LogisticRegression"
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.build_model()
        from sklearn.linear_model import LogisticRegression

        assert isinstance(model, LogisticRegression)

    def test_build_model_random_forest(self):
        """Test building Random Forest model through build_model method."""
        train_data, _ = self.train_data
        self.mock_config.data.model = "RandomForest"
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.build_model()
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(model, RandomForestClassifier)

    def test_build_model_xgboost(self):
        """Test building XGBoost model through build_model method."""
        train_data, _ = self.train_data
        self.mock_config.data.model = "XGBoost"
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        model = builder.build_model()
        import xgboost as xgb

        assert isinstance(model, xgb.XGBClassifier)

    def test_build_model_unsupported_type(self):
        """Test building model with unsupported type raises ValueError."""
        train_data, _ = self.train_data
        self.mock_config.data.model = "UnsupportedModel"
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        with pytest.raises(ValueError, match="Unsupported model type provided"):
            builder.build_model()

    @patch("common.models.model.PathManager")
    def test_prepare_directories(self, mock_path_manager):
        """Test directory preparation for model saving."""
        train_data, _ = self.train_data
        mock_path_manager.return_value.get_experiment_dir.return_value = (
            "/tmp/test_model"
        )

        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        with patch("os.makedirs") as mock_makedirs:
            model_dir = builder._prepare_directories()
            assert model_dir == "/tmp/test_model"
            mock_makedirs.assert_called_once_with("/tmp/test_model", exist_ok=True)

    def test_define_paths(self):
        """Test path definition for model artifacts."""
        train_data, _ = self.train_data
        builder = ModelBuilder(
            self.mock_config,
            train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
        )

        paths = builder._define_paths("/tmp/test_model")

        expected_paths = [
            "label_path",
            "reference_path",
            "scaler_path",
            "is_scaler_fit_path",
            "num_features_path",
            "highly_variable_genes_path",
            "mix_included_path",
            "history_path",
            "experiment_dir",
            "components_dir",
        ]

        for path_key in expected_paths:
            assert path_key in paths
            if path_key == "experiment_dir":
                assert paths[path_key] == "/tmp/test_model"
            elif path_key == "components_dir":
                assert paths[path_key] == "/tmp/test_model/model_components"
            else:
                assert paths[path_key].startswith("/tmp/test_model/")


class TestModelLoader:
    """Test ModelLoader functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()

    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        config.data = Mock()
        config.data.model = "CNN"
        return config

    @patch("common.models.model.PathManager")
    def test_model_loader_initialization(self, mock_path_manager):
        """Test ModelLoader initialization."""
        mock_path_manager.return_value.get_best_model_dir_for_config.return_value = (
            "/tmp/test_model"
        )

        loader = ModelLoader(self.mock_config)

        assert loader.config == self.mock_config
        assert loader.model_type == "cnn"
        assert loader.model_dir == "/tmp/test_model"

    def test_get_model_path_neural_network(self):
        """Test model path generation for neural networks."""
        with patch("common.models.model.PathManager"):
            loader = ModelLoader(self.mock_config)
            # Model type is set automatically from config
            loader.model_dir = "/tmp/test_model"

            path = loader._get_model_path()
            assert path == "/tmp/test_model/model.h5"

    def test_get_model_path_sklearn_model(self):
        """Test model path generation for sklearn models."""
        self.mock_config.data.model = "LogisticRegression"
        with patch("common.models.model.PathManager"):
            loader = ModelLoader(self.mock_config)
            # Model type is set automatically from config
            loader.model_dir = "/tmp/test_model"

            path = loader._get_model_path()
            assert path == "/tmp/test_model/model.pkl"

    @patch("os.path.exists")
    @patch("tensorflow.keras.models.load_model")
    def test_load_model_neural_network(self, mock_load_model, mock_exists):
        """Test loading neural network model."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        with patch("common.models.model.PathManager"):
            loader = ModelLoader(self.mock_config)
            # Model type is set automatically from config
            loader.model_path = "/tmp/test_model/best_model.h5"

            model = loader.load_model()
            assert model == mock_model
            mock_load_model.assert_called_once_with("/tmp/test_model/best_model.h5")

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("dill.load")
    def test_load_model_sklearn(self, mock_pickle_load, mock_file, mock_exists):
        """Test loading sklearn model."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model

        self.mock_config.data.model = "LogisticRegression"
        with patch("common.models.model.PathManager"):
            loader = ModelLoader(self.mock_config)
            # Model type is set automatically from config
            loader.model_path = "/tmp/test_model/model.pkl"

            model = loader.load_model()
            assert model == mock_model
            # Note: The loader also tries to read metadata.json, so we can't use assert_called_once_with
            # Instead, check that the model file was opened with correct path
            mock_file.assert_any_call("/tmp/test_model/model.pkl", "rb")


class TestCustomModelCheckpoint:
    """Test CustomModelCheckpoint functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.filepath = "/tmp/model.h5"
        self.best_val_loss_path = "/tmp/best_val_loss.json"
        self.label_path = "/tmp/label_encoder.pkl"
        self.label_encoder = Mock()
        self.reference_path = "/tmp/reference_data.npy"
        self.reference = np.random.randn(50, 100)
        self.scaler = Mock()
        self.scaler_path = "/tmp/scaler.pkl"
        self.is_scaler_fit = True
        self.is_scaler_fit_path = "/tmp/is_scaler_fit.pkl"
        self.highly_variable_genes = ["gene1", "gene2"]
        self.highly_variable_genes_path = "/tmp/hvg.pkl"
        self.mix_included = False
        self.mix_included_path = "/tmp/mix_included.pkl"
        self.num_features = 100
        self.num_features_path = "/tmp/num_features.pkl"

    def test_custom_model_checkpoint_initialization(self):
        """Test CustomModelCheckpoint initialization."""
        checkpoint = CustomModelCheckpoint(
            self.filepath,
            self.best_val_loss_path,
            self.label_path,
            self.label_encoder,
            self.reference_path,
            self.reference,
            self.scaler,
            self.scaler_path,
            self.is_scaler_fit,
            self.is_scaler_fit_path,
            self.highly_variable_genes,
            self.highly_variable_genes_path,
            self.mix_included,
            self.mix_included_path,
            self.num_features,
            self.num_features_path,
        )

        assert checkpoint.best_val_loss == float("inf")
        assert checkpoint.best_val_loss_path == self.best_val_loss_path
        assert checkpoint.label_encoder == self.label_encoder
        assert not checkpoint.model_improved

    def test_set_best_val_loss(self):
        """Test setting best validation loss."""
        checkpoint = CustomModelCheckpoint(
            self.filepath,
            self.best_val_loss_path,
            self.label_path,
            self.label_encoder,
            self.reference_path,
            self.reference,
            self.scaler,
            self.scaler_path,
            self.is_scaler_fit,
            self.is_scaler_fit_path,
            self.highly_variable_genes,
            self.highly_variable_genes_path,
            self.mix_included,
            self.mix_included_path,
            self.num_features,
            self.num_features_path,
        )

        checkpoint.set_best_val_loss(0.5)
        assert checkpoint.best_val_loss == 0.5
        assert checkpoint.best == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
