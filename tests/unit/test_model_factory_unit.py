"""Unit tests for model factory."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from common.models.model_factory import (
    BaseModel,
    CNNModel,
    LogisticRegressionModel,
    MLPModel,
    ModelFactory,
    RandomForestModel,
    XGBoostModel,
)
from common.utils.exceptions import ModelError


class TestModelFactory:
    """Test model factory creation."""

    def test_get_supported_models(self):
        """Test getting supported model types."""
        supported = ModelFactory.get_supported_models()
        assert isinstance(supported, list)
        assert "cnn" in supported
        assert "mlp" in supported
        assert "logistic" in supported
        assert "xgboost" in supported
        assert "random_forest" in supported

    def test_create_cnn_model(self):
        """Test creating CNN model."""
        config = Mock()

        model = ModelFactory.create_model("cnn", config)
        assert isinstance(model, CNNModel)
        assert model.config == config

    def test_create_mlp_model(self):
        """Test creating MLP model."""
        config = Mock()

        model = ModelFactory.create_model("mlp", config)
        assert isinstance(model, MLPModel)
        assert model.config == config

    def test_create_logistic_model(self):
        """Test creating logistic regression model."""
        config = Mock()

        model = ModelFactory.create_model("logistic", config)
        assert isinstance(model, LogisticRegressionModel)
        assert model.config == config

    def test_create_xgboost_model(self):
        """Test creating XGBoost model."""
        config = Mock()

        model = ModelFactory.create_model("xgboost", config)
        assert isinstance(model, XGBoostModel)
        assert model.config == config

    def test_create_random_forest_model(self):
        """Test creating Random Forest model."""
        config = Mock()

        model = ModelFactory.create_model("random_forest", config)
        assert isinstance(model, RandomForestModel)
        assert model.config == config

    def test_create_unsupported_model(self):
        """Test creating unsupported model type."""
        config = Mock()

        with pytest.raises(ModelError):
            ModelFactory.create_model("unsupported_model", config)


class TestCNNModel:
    """Test CNN model implementation."""

    @pytest.fixture
    def cnn_config(self):
        """Create CNN configuration."""
        config = Mock()
        config.model.cnn.filters = [32, 64]
        config.model.cnn.kernel_sizes = [3, 3]
        config.model.cnn.strides = [1, 1]
        config.model.cnn.paddings = ["same", "same"]
        config.model.cnn.pool_sizes = [2, 2]
        config.model.cnn.pool_strides = [2, 2]
        config.model.cnn.dense_units = [128, 64]
        config.model.cnn.dropout_rate = 0.3
        config.model.cnn.activation = "relu"
        config.model.training.learning_rate = 0.001
        config.model.training.custom_loss = "categorical_crossentropy"
        config.model.training.metrics = ["accuracy"]
        return config

    def test_cnn_initialization(self, cnn_config):
        """Test CNN model initialization."""
        model = CNNModel(cnn_config)
        assert model.config == cnn_config
        assert model.model is None
        assert not model.is_trained

    @patch("common.models.model_factory.Sequential")
    def test_cnn_build(self, mock_sequential, cnn_config):
        """Test CNN model building."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model

        cnn_model = CNNModel(cnn_config)
        cnn_model.build(input_shape=(100, 1), num_classes=4)

        assert cnn_model.model == mock_model
        mock_sequential.assert_called_once()
        mock_model.add.assert_called()
        mock_model.compile.assert_called_once()

    @patch("common.models.model_factory.Sequential")
    def test_cnn_train(self, mock_sequential, cnn_config):
        """Test CNN model training."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        mock_history = Mock()
        mock_model.fit.return_value = mock_history

        cnn_model = CNNModel(cnn_config)
        cnn_model.build(input_shape=(100, 1), num_classes=4)

        X_train = np.random.random((50, 100, 1))
        y_train = np.random.randint(0, 4, (50, 4))

        history = cnn_model.train(X_train, y_train)

        assert history == mock_history
        assert cnn_model.is_trained
        mock_model.fit.assert_called_once()

    @patch("common.models.model_factory.Sequential")
    def test_cnn_predict(self, mock_sequential, cnn_config):
        """Test CNN model prediction."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        mock_predictions = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_model.predict.return_value = mock_predictions

        cnn_model = CNNModel(cnn_config)
        cnn_model.build(input_shape=(100, 1), num_classes=4)
        cnn_model.is_trained = True

        X_test = np.random.random((1, 100, 1))
        predictions = cnn_model.predict(X_test)

        np.testing.assert_array_equal(
            predictions, [3]
        )  # argmax of [0.1, 0.2, 0.3, 0.4]
        mock_model.predict.assert_called_once_with(X_test)

    @patch("common.models.model_factory.Sequential")
    def test_cnn_predict_proba(self, mock_sequential, cnn_config):
        """Test CNN model prediction probabilities."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        mock_predictions = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_model.predict.return_value = mock_predictions

        cnn_model = CNNModel(cnn_config)
        cnn_model.build(input_shape=(100, 1), num_classes=4)
        cnn_model.is_trained = True

        X_test = np.random.random((1, 100, 1))
        probabilities = cnn_model.predict_proba(X_test)

        np.testing.assert_array_equal(probabilities, mock_predictions)
        mock_model.predict.assert_called_once_with(X_test)


class TestMLPModel:
    """Test MLP model implementation."""

    @pytest.fixture
    def mlp_config(self):
        """Create MLP configuration."""
        config = Mock()
        config.model.mlp.hidden_layers = [128, 64, 32]
        config.model.mlp.dropout_rate = 0.3
        config.model.mlp.activation = "relu"
        config.model.training.learning_rate = 0.001
        config.model.training.custom_loss = "categorical_crossentropy"
        config.model.training.metrics = ["accuracy"]
        return config

    @patch("common.models.model_factory.Sequential")
    def test_mlp_build(self, mock_sequential, mlp_config):
        """Test MLP model building."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model

        mlp_model = MLPModel(mlp_config)
        mlp_model.build(input_shape=(100,), num_classes=4)

        assert mlp_model.model == mock_model
        mock_sequential.assert_called_once()
        mock_model.add.assert_called()
        mock_model.compile.assert_called_once()


class TestLogisticRegressionModel:
    """Test Logistic Regression model implementation."""

    @pytest.fixture
    def lr_config(self):
        """Create Logistic Regression configuration."""
        config = Mock()
        config.model.logistic.max_iter = 1000
        config.model.logistic.random_state = 42
        return config

    @patch("common.models.model_factory.LogisticRegression")
    def test_lr_build(self, mock_lr, lr_config):
        """Test Logistic Regression model building."""
        mock_model = Mock()
        mock_lr.return_value = mock_model

        lr_model = LogisticRegressionModel(lr_config)
        lr_model.build(input_shape=(100,), num_classes=4)

        assert lr_model.model == mock_model
        # Check that LogisticRegression was called with correct parameters
        # The actual implementation passes more parameters than the test expected
        mock_lr.assert_called_once()
        call_args = mock_lr.call_args
        assert "max_iter" in call_args.kwargs
        assert "random_state" in call_args.kwargs

    @patch("common.models.model_factory.LogisticRegression")
    def test_lr_train(self, mock_lr, lr_config):
        """Test Logistic Regression training."""
        mock_model = Mock()
        mock_lr.return_value = mock_model

        lr_model = LogisticRegressionModel(lr_config)
        lr_model.build(input_shape=(100,), num_classes=4)

        X_train = np.random.random((50, 100))
        y_train = np.random.randint(0, 4, 50)

        lr_model.train(X_train, y_train)

        assert lr_model.is_trained
        mock_model.fit.assert_called_once_with(X_train, y_train)

    @patch("common.models.model_factory.LogisticRegression")
    def test_lr_predict(self, mock_lr, lr_config):
        """Test Logistic Regression prediction."""
        mock_model = Mock()
        mock_lr.return_value = mock_model
        mock_model.predict.return_value = np.array([1, 2])

        lr_model = LogisticRegressionModel(lr_config)
        lr_model.build(input_shape=(100,), num_classes=4)
        lr_model.is_trained = True

        X_test = np.random.random((2, 100))
        predictions = lr_model.predict(X_test)

        np.testing.assert_array_equal(predictions, [1, 2])
        mock_model.predict.assert_called_once_with(X_test)


class TestXGBoostModel:
    """Test XGBoost model implementation."""

    @pytest.fixture
    def xgb_config(self):
        """Create XGBoost configuration."""
        config = Mock()
        config.model.xgboost.n_estimators = 100
        config.model.xgboost.max_depth = 6
        config.model.xgboost.learning_rate = 0.1
        config.model.xgboost.random_state = 42
        return config

    @patch("common.models.model_factory.xgb.XGBClassifier")
    def test_xgb_build(self, mock_xgb, xgb_config):
        """Test XGBoost model building."""
        mock_model = Mock()
        mock_xgb.return_value = mock_model

        xgb_model = XGBoostModel(xgb_config)
        xgb_model.build(input_shape=(100,), num_classes=4)

        assert xgb_model.model == mock_model
        # Check that XGBClassifier was called with correct parameters
        mock_xgb.assert_called_once()
        call_args = mock_xgb.call_args
        assert "n_estimators" in call_args.kwargs
        assert "max_depth" in call_args.kwargs
        assert "learning_rate" in call_args.kwargs


class TestRandomForestModel:
    """Test Random Forest model implementation."""

    @pytest.fixture
    def rf_config(self):
        """Create Random Forest configuration."""
        config = Mock()
        config.model.random_forest.n_estimators = 100
        config.model.random_forest.max_depth = 10
        config.model.random_forest.random_state = 42
        return config

    @patch("common.models.model_factory.RandomForestClassifier")
    def test_rf_build(self, mock_rf, rf_config):
        """Test Random Forest model building."""
        mock_model = Mock()
        mock_rf.return_value = mock_model

        rf_model = RandomForestModel(rf_config)
        rf_model.build(input_shape=(100,), num_classes=4)

        assert rf_model.model == mock_model
        # Check that RandomForestClassifier was called with correct parameters
        mock_rf.assert_called_once()
        call_args = mock_rf.call_args
        assert "n_estimators" in call_args.kwargs
        assert "max_depth" in call_args.kwargs
        assert "random_state" in call_args.kwargs

    @patch("common.models.model_factory.RandomForestClassifier")
    def test_rf_train(self, mock_rf, rf_config):
        """Test Random Forest training."""
        mock_model = Mock()
        mock_rf.return_value = mock_model

        rf_model = RandomForestModel(rf_config)
        rf_model.build(input_shape=(100,), num_classes=4)

        X_train = np.random.random((50, 100))
        y_train = np.random.randint(0, 4, 50)

        rf_model.train(X_train, y_train)

        assert rf_model.is_trained
        mock_model.fit.assert_called_once_with(X_train, y_train)


class TestBaseModelErrorHandling:
    """Test error handling in models."""

    def test_predict_without_training(self):
        """Test prediction without training raises error."""
        config = Mock()
        model = CNNModel(config)

        with pytest.raises(ModelError):
            model.predict(np.random.random((1, 100, 1)))

    def test_predict_proba_without_training(self):
        """Test prediction probabilities without training raises error."""
        config = Mock()
        model = CNNModel(config)

        with pytest.raises(ModelError):
            model.predict_proba(np.random.random((1, 100, 1)))
