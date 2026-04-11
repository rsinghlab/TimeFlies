"""Unit tests for model factory."""

from unittest.mock import Mock

import numpy as np
import pytest

from timeflies.models.model_factory import (
    CNNModel,
    LogisticRegressionModel,
    MLPModel,
    ModelFactory,
    RandomForestModel,
    XGBoostModel,
)
from timeflies.utils.exceptions import ModelError


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
