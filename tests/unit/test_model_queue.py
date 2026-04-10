"""Tests for model queue functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from timeflies.core.model_queue import ModelQueueManager


class TestModelQueueManager:
    """Test the ModelQueueManager class."""

    @pytest.fixture
    def sample_queue_config(self):
        """Create a sample queue configuration."""
        return {
            "queue_settings": {
                "name": "test_queue",
                "sequential": True,
                "save_checkpoints": True,
                "generate_summary": True,
            },
            "model_queue": [
                {
                    "name": "test_cnn",
                    "model_type": "CNN",
                    "description": "Test CNN model",
                    "hyperparameters": {"epochs": 5, "batch_size": 32},
                },
                {
                    "name": "test_xgboost",
                    "model_type": "xgboost",
                    "description": "Test XGBoost model",
                    "hyperparameters": {"n_estimators": 10, "max_depth": 3},
                    "config_overrides": {
                        "data": {"batch_correction": {"enabled": True}}
                    },
                },
            ],
            "global_settings": {
                "project": "fruitfly_aging",
                "data": {
                    "tissue": "head",
                    "target_variable": "age",
                    "batch_correction": {"enabled": False},
                },
                "with_eda": False,
                "with_analysis": True,
            },
        }

    @pytest.fixture
    def temp_config_file(self, sample_queue_config):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_queue_config, f)
            return f.name

    def test_load_queue_config(self, temp_config_file):
        """Test loading queue configuration from file."""
        manager = ModelQueueManager(temp_config_file)

        assert manager.queue_settings["name"] == "test_queue"
        assert len(manager.model_queue) == 2
        assert manager.model_queue[0]["name"] == "test_cnn"
        assert manager.model_queue[1]["name"] == "test_xgboost"

        # Clean up
        Path(temp_config_file).unlink()

    def test_load_queue_config_missing_file(self):
        """Test error handling for missing configuration file."""
        with pytest.raises(FileNotFoundError, match="Queue config not found"):
            ModelQueueManager("nonexistent.yaml")

    def test_load_queue_config_invalid_structure(self):
        """Test error handling for invalid configuration structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"invalid": "config"}, f)
            temp_file = f.name

        with pytest.raises(ValueError, match="must contain 'model_queue' section"):
            ModelQueueManager(temp_file)

        Path(temp_file).unlink()

    def test_deep_merge_dict(self, temp_config_file):
        """Test deep merging of configuration dictionaries."""
        manager = ModelQueueManager(temp_config_file)

        base = {
            "data": {"batch_correction": {"enabled": False}, "tissue": "head"},
            "model": {"epochs": 100},
        }

        override = {
            "data": {"batch_correction": {"enabled": True}},
            "model": {"batch_size": 32},
        }

        result = manager._deep_merge_dict(base, override)

        # Should merge nested dictionaries
        assert result["data"]["batch_correction"]["enabled"] is True
        assert result["data"]["tissue"] == "head"  # Preserved from base
        assert result["model"]["epochs"] == 100  # Preserved from base
        assert result["model"]["batch_size"] == 32  # Added from override

        # Clean up
        Path(temp_config_file).unlink()

    def test_prepare_model_config(self, temp_config_file):
        """Test preparation of model configuration with overrides."""
        manager = ModelQueueManager(temp_config_file)

        model_config = {
            "name": "test_model",
            "model_type": "CNN",
            "hyperparameters": {"epochs": 50, "batch_size": 16},
            "config_overrides": {"data": {"batch_correction": {"enabled": True}}},
        }

        config = manager.prepare_model_config(model_config)

        # Check model type is set in data section
        assert config["data"]["model"] == "CNN"
        assert config["experiment_name"] == "test_model"

        # Check global settings are applied
        assert config["project"] == "fruitfly_aging"
        assert config["data"]["tissue"] == "head"

        # Check overrides are applied
        assert config["data"]["batch_correction"]["enabled"] is True

        # Clean up
        Path(temp_config_file).unlink()

    @patch("timeflies.core.model_queue.train_command")
    @patch("timeflies.core.model_queue.evaluate_command")
    def test_train_single_model_success(
        self, mock_evaluate, mock_train, temp_config_file
    ):
        """Test successful training of a single model."""
        manager = ModelQueueManager(temp_config_file)

        model_config = {
            "name": "test_cnn",
            "model_type": "CNN",
            "description": "Test CNN",
            "hyperparameters": {"epochs": 5, "batch_size": 32},
        }

        # Mock successful training (train_command includes evaluation internally)
        mock_train.return_value = 0
        mock_evaluate.return_value = 0

        result = manager.train_single_model(0, model_config)

        # Check result structure
        assert result["name"] == "test_cnn"
        assert result["model_type"] == "CNN"
        assert result["status"] == "completed"
        assert "training_time" in result
        assert "timestamp" in result

        # train_command is called (it handles evaluation internally)
        mock_train.assert_called_once()

        # Clean up
        Path(temp_config_file).unlink()

    @patch("timeflies.core.model_queue.train_command")
    def test_train_single_model_failure(self, mock_train, temp_config_file):
        """Test handling of training failures."""
        manager = ModelQueueManager(temp_config_file)

        model_config = {
            "name": "test_cnn",
            "model_type": "CNN",
            "description": "Test CNN",
            "hyperparameters": {"epochs": 5, "batch_size": 32},
        }

        # Mock training failure
        mock_train.side_effect = Exception("Training failed")

        result = manager.train_single_model(0, model_config)

        # Check error handling
        assert result["name"] == "test_cnn"
        assert result["status"] == "failed"
        assert "Training failed" in result["error"]

        # Clean up
        Path(temp_config_file).unlink()

    @patch("timeflies.core.model_queue.ModelQueueManager.train_single_model")
    def test_run_queue(self, mock_train_single, temp_config_file):
        """Test running the complete queue."""
        manager = ModelQueueManager(temp_config_file)

        # Mock successful training results
        mock_train_single.side_effect = [
            {
                "name": "test_cnn",
                "status": "completed",
                "training_time": 10.5,
                "metrics": {"accuracy": 0.85},
            },
            {
                "name": "test_xgboost",
                "status": "completed",
                "training_time": 5.2,
                "metrics": {"accuracy": 0.82},
            },
        ]

        # Mock the generate_summary_report method
        with patch.object(manager, "generate_summary_report") as mock_generate_summary:
            manager.run_queue(resume=False)

            # Verify all models were trained
            assert mock_train_single.call_count == 2
            mock_generate_summary.assert_called_once()

            # Check results were stored
            assert len(manager.results) == 2
            assert manager.results[0]["name"] == "test_cnn"
            assert manager.results[1]["name"] == "test_xgboost"

        # Clean up
        Path(temp_config_file).unlink()

    def test_checkpoint_save_load(self, temp_config_file):
        """Test checkpoint saving and loading functionality."""
        manager = ModelQueueManager(temp_config_file)

        # Add some mock results
        manager.results = [
            {"name": "model1", "status": "completed"},
            {"name": "model2", "status": "completed"},
        ]
        manager.completed_models = ["model1", "model2"]

        # Save checkpoint
        manager.save_checkpoint(1)

        # Create new manager and load checkpoint
        manager2 = ModelQueueManager(temp_config_file)
        checkpoint_index = manager2.load_checkpoint()

        assert checkpoint_index == 1
        assert len(manager2.results) == 2
        assert manager2.results[0]["name"] == "model1"
        assert "model1" in manager2.completed_models

        # Clean up
        Path(temp_config_file).unlink()
        if manager.checkpoint_file.exists():
            manager.checkpoint_file.unlink()

    def test_generate_summary_report(self, temp_config_file):
        """Test summary report generation."""
        manager = ModelQueueManager(temp_config_file)

        manager.results = [
            {
                "name": "cnn_model",
                "model_type": "CNN",
                "description": "Test CNN model",
                "status": "completed",
                "training_time": 10.5,
                "hyperparameters": {"epochs": 50, "batch_size": 32},
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.88,
                    "f1_score": 0.85,
                },
            },
        ]

        manager.start_time = 1234567890

        with tempfile.TemporaryDirectory() as temp_dir:
            # Point outputs to temp dir so report files are created there
            with patch("timeflies.core.model_queue.Path") as mock_path_cls:
                # Make Path("outputs") return our temp dir
                real_path = Path
                def path_side_effect(p):
                    if str(p) == "outputs":
                        return real_path(temp_dir) / "outputs"
                    return real_path(p)
                mock_path_cls.side_effect = path_side_effect

                report_path, csv_path = manager.generate_summary_report()

                assert report_path is not None
                assert csv_path is not None

        Path(temp_config_file).unlink()
