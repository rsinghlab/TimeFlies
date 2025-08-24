"""Unit tests for PipelineManager core functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from common.core import PipelineManager


class TestPipelineManager:
    """Test PipelineManager core functionality."""

    def test_pipeline_manager_initialization(self):
        """Test PipelineManager initialization."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.project = "fruitfly_aging"
        config.general.random_state = 42

        pipeline = PipelineManager(config)
        assert pipeline.config == config
        assert pipeline.random_state == 42

    def test_pipeline_manager_initialization_with_defaults(self):
        """Test PipelineManager initialization with default values."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.project = "fruitfly_aging"
        # No random_state defined
        del config.general.random_state

        pipeline = PipelineManager(config)
        assert pipeline.random_state == 42  # Should use default

    def test_load_data_pipeline(self, small_sample_anndata):
        """Test data loading pipeline."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.project = "fruitfly_aging"

        pipeline = PipelineManager(config)

        with patch.object(
            pipeline.data_processor, "load_data", return_value=small_sample_anndata
        ):
            adata = pipeline.load_data()
            assert adata is not None
            assert adata.n_obs > 0
            assert adata.n_vars > 0

    def test_preprocess_pipeline(self, small_sample_anndata):
        """Test preprocessing pipeline."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.batch_correction.enabled = False

        pipeline = PipelineManager(config)

        with patch.object(
            pipeline.data_processor,
            "preprocess_data",
            return_value=small_sample_anndata,
        ):
            with patch.object(
                pipeline.gene_filter,
                "apply_gene_filtering",
                return_value=small_sample_anndata,
            ):
                processed = pipeline.preprocess_data(small_sample_anndata.copy())
                assert processed is not None

    def test_preprocess_pipeline_with_batch_correction(self, small_sample_anndata):
        """Test preprocessing pipeline with batch correction."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.batch_correction.enabled = True

        pipeline = PipelineManager(config)

        with patch.object(
            pipeline.data_processor,
            "preprocess_data",
            return_value=small_sample_anndata,
        ):
            with patch.object(
                pipeline.gene_filter,
                "apply_gene_filtering",
                return_value=small_sample_anndata,
            ):
                with patch.object(
                    pipeline.batch_corrector,
                    "apply_batch_correction",
                    return_value=small_sample_anndata,
                ):
                    processed = pipeline.preprocess_data(small_sample_anndata.copy())
                    assert processed is not None

    def test_split_data_pipeline(self, small_sample_anndata):
        """Test data splitting pipeline."""
        config = MagicMock()
        config.data.split.method = "random"
        config.data.split.test_size = 0.2

        pipeline = PipelineManager(config)

        train_data = small_sample_anndata[:8].copy()
        test_data = small_sample_anndata[8:].copy()

        with patch.object(
            pipeline.data_processor, "split_data", return_value=(train_data, test_data)
        ):
            train, test = pipeline.split_data(small_sample_anndata)
            assert train is not None
            assert test is not None
            assert train.n_obs + test.n_obs == small_sample_anndata.n_obs

    def test_prepare_model_data(self, small_sample_anndata):
        """Test model data preparation."""
        config = MagicMock()
        config.data.target_variable = "age"

        pipeline = PipelineManager(config)

        X, y = pipeline.prepare_model_data(small_sample_anndata)

        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X) == small_sample_anndata.n_obs

    def test_create_model(self):
        """Test model creation."""
        config = MagicMock()
        config.model.type = "CNN"
        config.model.architecture.layers = [64, 32]
        config.model.architecture.dropout = 0.2

        pipeline = PipelineManager(config)

        with patch.object(pipeline.model_factory, "create_model") as mock_create:
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            model = pipeline.create_model(input_shape=(100,), num_classes=3)

            assert model is not None
            mock_create.assert_called_once()

    def test_train_model(self, small_sample_anndata):
        """Test model training."""
        config = MagicMock()
        config.model.training.epochs = 2
        config.model.training.batch_size = 16

        pipeline = PipelineManager(config)

        # Mock model and training data
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_model.fit.return_value = mock_history

        X = np.random.rand(10, 100)
        y = np.random.randint(0, 3, 10)

        history = pipeline.train_model(mock_model, X, y)

        assert history is not None
        mock_model.fit.assert_called_once()

    def test_evaluate_model(self):
        """Test model evaluation."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        # Mock model and test data
        mock_model = MagicMock()
        mock_model.evaluate.return_value = [0.5, 0.8]  # loss, accuracy
        mock_model.predict.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

        X_test = np.random.rand(2, 100)
        y_test = np.array([1, 0])

        metrics = pipeline.evaluate_model(mock_model, X_test, y_test)

        assert metrics is not None
        assert "loss" in metrics
        assert "accuracy" in metrics
        mock_model.evaluate.assert_called_once()
        mock_model.predict.assert_called_once()

    def test_save_model(self):
        """Test model saving."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")

            with patch.object(mock_model, "save") as mock_save:
                pipeline.save_model(mock_model, model_path)
                mock_save.assert_called_once_with(model_path)

    def test_load_model(self):
        """Test model loading."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")

            # Create a mock saved model directory
            os.makedirs(model_path)

            with patch("tensorflow.keras.models.load_model") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                loaded_model = pipeline.load_model(model_path)

                assert loaded_model is not None
                mock_load.assert_called_once_with(model_path)

    def test_run_full_pipeline(self, small_sample_anndata):
        """Test running the full pipeline."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.target_variable = "age"
        config.model.type = "CNN"
        config.model.training.epochs = 1

        pipeline = PipelineManager(config)

        # Mock all the components
        with patch.object(pipeline, "load_data", return_value=small_sample_anndata):
            with patch.object(
                pipeline, "preprocess_data", return_value=small_sample_anndata
            ):
                with patch.object(
                    pipeline,
                    "split_data",
                    return_value=(small_sample_anndata[:8], small_sample_anndata[8:]),
                ):
                    with patch.object(
                        pipeline,
                        "prepare_model_data",
                        return_value=(
                            np.random.rand(8, 100),
                            np.random.randint(0, 3, 8),
                        ),
                    ):
                        with patch.object(
                            pipeline, "create_model", return_value=MagicMock()
                        ):
                            with patch.object(
                                pipeline, "train_model", return_value=MagicMock()
                            ):
                                with patch.object(
                                    pipeline,
                                    "evaluate_model",
                                    return_value={"accuracy": 0.8},
                                ):
                                    result = pipeline.run_full_pipeline()

                                    assert result is not None
                                    assert "model" in result
                                    assert "metrics" in result

    def test_get_pipeline_status(self):
        """Test pipeline status reporting."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        status = pipeline.get_pipeline_status()

        assert isinstance(status, dict)
        assert "data_loaded" in status
        assert "model_trained" in status
        assert "preprocessing_complete" in status

    def test_reset_pipeline(self):
        """Test pipeline reset functionality."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        # Set some state
        pipeline.data_loaded = True
        pipeline.model_trained = True

        pipeline.reset_pipeline()

        assert pipeline.data_loaded == False
        assert pipeline.model_trained == False

    def test_validate_config(self):
        """Test configuration validation."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.target_variable = "age"
        config.model.type = "CNN"

        pipeline = PipelineManager(config)

        # Should not raise an exception
        is_valid = pipeline.validate_config()
        assert isinstance(is_valid, bool)

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        config = MagicMock()
        # Missing required fields
        del config.data.tissue

        pipeline = PipelineManager(config)

        with pytest.raises(AttributeError):
            pipeline.validate_config()

    def test_set_random_seed(self):
        """Test random seed setting."""
        config = MagicMock()
        config.general.random_state = 123

        pipeline = PipelineManager(config)

        with patch("numpy.random.seed") as mock_np_seed:
            with patch("tensorflow.random.set_seed") as mock_tf_seed:
                pipeline.set_random_seed()

                mock_np_seed.assert_called_with(123)
                mock_tf_seed.assert_called_with(123)

    def test_log_pipeline_step(self):
        """Test pipeline step logging."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        with patch.object(pipeline, "logger") as mock_logger:
            pipeline.log_pipeline_step("test_step", "completed")
            mock_logger.info.assert_called()

    def test_handle_pipeline_error(self):
        """Test pipeline error handling."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        with patch.object(pipeline, "logger") as mock_logger:
            test_error = ValueError("Test error")
            pipeline.handle_pipeline_error("test_step", test_error)
            mock_logger.error.assert_called()

    def test_cleanup_pipeline(self):
        """Test pipeline cleanup."""
        config = MagicMock()
        pipeline = PipelineManager(config)

        # Set some temporary state
        pipeline.temp_data = "some_data"

        pipeline.cleanup_pipeline()

        # Should reset temporary state
        assert not hasattr(pipeline, "temp_data")

    def test_get_pipeline_config_summary(self):
        """Test pipeline configuration summary."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.target_variable = "age"
        config.model.type = "CNN"

        pipeline = PipelineManager(config)

        summary = pipeline.get_config_summary()

        assert isinstance(summary, dict)
        assert "tissue" in summary
        assert "target_variable" in summary
        assert "model_type" in summary
