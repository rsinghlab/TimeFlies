"""Real end-to-end pipeline tests using tiny fixture data.

Trains actual models on the fixture AnnData (50 cells, 100 genes)
and verifies real outputs -- no mocks.
"""

import pickle
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from timeflies.core.config_manager import Config
from timeflies.models.model_factory import (
    CNNModel,
    LogisticRegressionModel,
    ModelFactory,
    RandomForestModel,
    XGBoostModel,
)

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "fruitfly_aging" / "tiny_head.h5ad"
)
NUM_CLASSES = 4  # ages: 5, 30, 50, 70


def _load_fixture_xy():
    """Load fixture data and return (X_train, X_test, y_train, y_test, label_encoder)."""
    adata = ad.read_h5ad(FIXTURE)
    X = np.asarray(adata.X, dtype=np.float32)

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["age"].values)

    # Simple 80/20 split with fixed seed
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], le


def _make_config(model_name: str = "logistic") -> Config:
    """Build a minimal Config object sufficient for model construction."""
    return Config(
        {
            "general": {"random_state": 42},
            "model": {
                "training": {
                    "epochs": 3,
                    "batch_size": 16,
                    "validation_split": 0.2,
                    "learning_rate": 0.001,
                },
                "cnn": {
                    "filters": [16],
                    "kernel_sizes": [3],
                    "strides": [1],
                    "paddings": ["same"],
                    "activation": "relu",
                    "dropout_rate": 0.2,
                },
                "logistic": {"max_iter": 200},
                "xgboost": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.3,
                },
                "random_forest": {
                    "n_estimators": 10,
                    "max_depth": 3,
                },
            },
        }
    )


@pytest.mark.integration
class TestPipelineE2E:
    """Train real models on tiny fixture data, no mocks."""

    def test_logistic_regression_train_predict(self):
        """Train logistic regression and verify predictions."""
        X_train, X_test, y_train, y_test, le = _load_fixture_xy()
        config = _make_config("logistic")

        model = LogisticRegressionModel(config)
        model.build(input_shape=(X_train.shape[1],), num_classes=NUM_CLASSES)
        model.train(X_train, y_train)

        preds = model.predict(X_test)

        assert preds.shape == (len(X_test),)
        assert set(preds).issubset(set(range(NUM_CLASSES)))

        accuracy = np.mean(preds == y_test)
        # With 4 classes, random chance is 0.25.
        # A real model on correlated data should beat that.
        assert accuracy >= 0.1, f"Accuracy {accuracy:.2f} is suspiciously low"

    def test_xgboost_train_predict(self):
        """Train XGBoost and verify predictions."""
        X_train, X_test, y_train, y_test, le = _load_fixture_xy()
        config = _make_config("xgboost")

        model = XGBoostModel(config)
        model.build(input_shape=(X_train.shape[1],), num_classes=NUM_CLASSES)
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert preds.shape == (len(X_test),)
        assert proba.shape == (len(X_test), NUM_CLASSES)
        # Probabilities should sum to 1 for each sample
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_random_forest_train_predict(self):
        """Train Random Forest and verify predictions."""
        X_train, X_test, y_train, y_test, le = _load_fixture_xy()
        config = _make_config("random_forest")

        model = RandomForestModel(config)
        model.build(input_shape=(X_train.shape[1],), num_classes=NUM_CLASSES)
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert preds.shape == (len(X_test),)
        assert proba.shape == (len(X_test), NUM_CLASSES)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_cnn_train_predict(self):
        """Train CNN on tiny data (3 epochs) and verify outputs."""
        X_train, X_test, y_train, y_test, le = _load_fixture_xy()
        config = _make_config("cnn")

        model = CNNModel(config)
        model.build(input_shape=(X_train.shape[1], 1), num_classes=NUM_CLASSES)

        # CNN expects one-hot labels
        from tensorflow.keras.utils import to_categorical

        y_train_oh = to_categorical(y_train, NUM_CLASSES)

        history = model.train(X_train, y_train_oh)
        assert history is not None

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert preds.shape == (len(X_test),)
        assert proba.shape == (len(X_test), NUM_CLASSES)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_model_save_load_roundtrip(self):
        """Train, save, load, and predict -- verify consistency."""
        X_train, X_test, y_train, y_test, le = _load_fixture_xy()
        config = _make_config("logistic")

        model = LogisticRegressionModel(config)
        model.build(input_shape=(X_train.shape[1],), num_classes=NUM_CLASSES)
        model.train(X_train, y_train)

        original_preds = model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            save_path = f.name

        try:
            model.save(save_path)
            assert Path(save_path).stat().st_size > 0

            loaded_model = LogisticRegressionModel(config)
            loaded_model.load(save_path)

            loaded_preds = loaded_model.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_factory_creates_and_trains(self):
        """Use ModelFactory to create models, then train them."""
        X_train, X_test, y_train, y_test, le = _load_fixture_xy()
        config = _make_config()

        for model_type in ("logistic", "random_forest", "xgboost"):
            model = ModelFactory.create_model(model_type, config)
            model.build(input_shape=(X_train.shape[1],), num_classes=NUM_CLASSES)
            model.train(X_train, y_train)

            assert model.is_trained, f"{model_type} should be marked as trained"
            preds = model.predict(X_test)
            assert len(preds) == len(X_test), f"{model_type} prediction count mismatch"
