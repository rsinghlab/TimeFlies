"""Tests for hyperparameter tuning functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from common.core.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner:
    """Test the HyperparameterTuner class."""

    @pytest.fixture
    def sample_config_with_tuning(self):
        """Create a sample configuration with hyperparameter tuning enabled."""
        return {
            # Basic project settings
            "project": "fruitfly_aging",
            "data": {"model": "CNN", "tissue": "head", "target_variable": "age"},
            # Hyperparameter tuning configuration
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "grid",
                "n_trials": 4,
                "search_optimizations": {
                    "data": {"sampling": {"samples": 100, "variables": 50}},
                    "with_eda": False,
                    "with_analysis": False,
                },
                "model_hyperparams": {
                    "CNN": {
                        "learning_rate": [0.001, 0.01],
                        "batch_size": [16, 32],
                        "cnn_variants": [
                            {"name": "standard", "filters": [32], "kernel_sizes": [3]},
                            {"name": "larger", "filters": [64], "kernel_sizes": [3]},
                        ],
                    },
                    "xgboost": {"n_estimators": [100, 200], "max_depth": [6, 9]},
                },
            },
            # Regular model configuration
            "model": {
                "training": {"epochs": 100, "batch_size": 32},
                "cnn": {"filters": [32], "kernel_sizes": [3]},
            },
        }

    @pytest.fixture
    def sample_config_disabled(self):
        """Create a sample configuration with hyperparameter tuning disabled."""
        return {
            "project": "fruitfly_aging",
            "data": {"model": "CNN"},
            "hyperparameter_tuning": {"enabled": False},
        }

    @pytest.fixture
    def temp_config_file(self, sample_config_with_tuning):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_with_tuning, f)
            return f.name

    def test_init_with_tuning_enabled(self, temp_config_file):
        """Test initialization with hyperparameter tuning enabled."""
        tuner = HyperparameterTuner(temp_config_file)

        assert tuner.search_method == "grid"
        assert tuner.n_trials == 4
        assert tuner.current_model_type == "CNN"

        # Clean up
        Path(temp_config_file).unlink()

    def test_init_with_tuning_disabled(self):
        """Test error when hyperparameter tuning is disabled."""
        config_disabled = {
            "project": "fruitfly_aging",
            "hyperparameter_tuning": {"enabled": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_disabled, f)
            temp_file = f.name

        with pytest.raises(ValueError, match="Hyperparameter tuning is not enabled"):
            HyperparameterTuner(temp_file)

        Path(temp_file).unlink()

    def test_generate_cnn_parameter_combinations(self, temp_config_file):
        """Test generation of CNN parameter combinations."""
        tuner = HyperparameterTuner(temp_config_file)

        combinations = tuner.generate_parameter_combinations()

        # Should generate combinations for both CNN variants × hyperparameters
        # 2 variants × 2 learning_rates × 2 batch_sizes = 8 combinations
        assert len(combinations) == 8

        # Check that all combinations have required fields
        for combo in combinations:
            assert "variant_name" in combo
            assert "model_type" in combo
            assert combo["model_type"] == "CNN"
            assert "config_overrides" in combo
            assert "hyperparameters" in combo

        # Check that CNN variants are properly represented
        variant_names = {combo["variant_name"] for combo in combinations}
        assert "standard" in variant_names
        assert "larger" in variant_names

        # Clean up
        Path(temp_config_file).unlink()

    def test_prepare_model_config(self, temp_config_file):
        """Test preparation of model configuration for a trial."""
        tuner = HyperparameterTuner(temp_config_file)

        trial_params = {
            "variant_name": "test_variant",
            "model_type": "CNN",
            "config_overrides": {"model": {"cnn": {"filters": [64]}}},
            "hyperparameters": {"learning_rate": 0.01, "batch_size": 32},
        }

        config, hyperparams = tuner.prepare_model_config(trial_params)

        # Check that base config is preserved
        assert config["project"] == "fruitfly_aging"
        assert config["data"]["model"] == "CNN"

        # Check that search optimizations are applied
        assert config["data"]["sampling"]["samples"] == 100
        assert config["with_eda"] is False

        # Check that config overrides are applied
        assert config["model"]["cnn"]["filters"] == [64]

        # Check that hyperparameters are extracted
        assert hyperparams["learning_rate"] == 0.01
        assert hyperparams["batch_size"] == 32

        # Check that hyperparameter_tuning section is removed
        assert "hyperparameter_tuning" not in config

        # Clean up
        Path(temp_config_file).unlink()

    def test_generate_traditional_ml_combinations(self):
        """Test generation of combinations for traditional ML models."""
        config_xgboost = {
            "project": "fruitfly_aging",
            "data": {"model": "xgboost"},
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "grid",
                "model_hyperparams": {
                    "xgboost": {"n_estimators": [100, 200], "max_depth": [6, 9]}
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_xgboost, f)
            temp_file = f.name

        tuner = HyperparameterTuner(temp_file)
        combinations = tuner.generate_parameter_combinations()

        # Should generate 2 × 2 = 4 combinations
        assert len(combinations) == 4

        # All should be xgboost
        for combo in combinations:
            assert combo["model_type"] == "xgboost"
            assert combo["variant_name"] == "xgboost_tuned"

        # Clean up
        Path(temp_file).unlink()

    @patch.dict(os.environ, {"PYTEST_CURRENT_TEST": ""}, clear=False)
    @patch("common.core.hyperparameter_tuner.time.time")
    def test_checkpoint_functionality(self, mock_time, temp_config_file):
        """Test checkpoint saving and loading with temporary directory."""
        mock_time.return_value = 1000000000

        # Create temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            tuner = HyperparameterTuner(temp_config_file)
            # Override the run_dir to use our temp directory
            tuner.run_dir = Path(temp_dir) / "test_run"
            tuner.checkpoint_file = tuner.run_dir / "checkpoint.json"

            # Add some mock results
            tuner.results = [
                {"name": "test1", "status": "completed", "metrics": {"accuracy": 0.85}},
                {"name": "test2", "status": "completed", "metrics": {"accuracy": 0.82}},
            ]
            tuner.completed_trials = [{"name": "test1"}, {"name": "test2"}]
            tuner.start_time = 1000000000

            # Save checkpoint (will create temp directory)
            tuner.save_checkpoint(1)

            # Create new tuner and load checkpoint
            tuner2 = HyperparameterTuner(temp_config_file)
            tuner2.checkpoint_file = tuner.checkpoint_file  # Use same checkpoint file

            checkpoint_index = tuner2.load_checkpoint()

            assert checkpoint_index == 1
            assert len(tuner2.results) == 2
            assert tuner2.results[0]["name"] == "test1"
            assert tuner2.start_time == 1000000000

        # Clean up
        Path(temp_config_file).unlink()
        if tuner.checkpoint_file and tuner.checkpoint_file.exists():
            tuner.checkpoint_file.unlink()
        if tuner.run_dir and tuner.run_dir.exists():
            import shutil

            shutil.rmtree(tuner.run_dir)

    def test_random_sampling(self, temp_config_file):
        """Test random sampling of parameter combinations."""
        tuner = HyperparameterTuner(temp_config_file)
        tuner.search_method = "random"
        tuner.n_trials = 3  # Less than total combinations (8)

        combinations = tuner.generate_parameter_combinations()

        # Should sample only 3 combinations
        assert len(combinations) == 3

        # All should be valid combinations
        for combo in combinations:
            assert "variant_name" in combo
            assert "model_type" in combo
            assert combo["model_type"] == "CNN"

        # Clean up
        Path(temp_config_file).unlink()

    def test_missing_hyperparameters_validation(self):
        """Test validation when no hyperparameters are defined for model type."""
        config_no_params = {
            "project": "fruitfly_aging",
            "data": {"model": "MLP"},  # MLP not defined in hyperparameters
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "grid",
                "model_hyperparams": {
                    "CNN": {"learning_rate": [0.01]}  # Only CNN defined
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_no_params, f)
            temp_file = f.name

        with pytest.raises(
            ValueError, match="No hyperparameters defined for model type: MLP"
        ):
            HyperparameterTuner(temp_file)

        Path(temp_file).unlink()

    def test_deep_merge_dict(self, temp_config_file):
        """Test deep merging of configuration dictionaries."""
        tuner = HyperparameterTuner(temp_config_file)

        base = {
            "data": {"sampling": {"samples": 1000, "variables": 500}, "tissue": "head"},
            "model": {"epochs": 100},
        }

        override = {
            "data": {
                "sampling": {"samples": 2000},  # Override samples but keep variables
                "new_field": "value",
            },
            "new_section": {"key": "value"},
        }

        result = tuner._deep_merge_dict(base, override)

        # Check that override values are applied
        assert result["data"]["sampling"]["samples"] == 2000

        # Check that non-overridden values are preserved
        assert result["data"]["sampling"]["variables"] == 500
        assert result["data"]["tissue"] == "head"

        # Check that new fields are added
        assert result["data"]["new_field"] == "value"
        assert result["new_section"]["key"] == "value"

        # Check that base model section is preserved
        assert result["model"]["epochs"] == 100

        # Clean up
        Path(temp_config_file).unlink()
