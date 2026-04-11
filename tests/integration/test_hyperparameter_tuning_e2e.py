"""End-to-end test for hyperparameter tuning functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from timeflies.core.hyperparameter_tuner import HyperparameterTuner


def create_minimal_tuning_config():
    """Create a minimal config for e2e hyperparameter tuning test."""
    return {
        "project": "fruitfly_aging",
        "data": {
            "model": "CNN",
            "tissue": "head",
            "target_variable": "age",
            "sampling": {"samples": 50, "variables": 25},
        },
        "hyperparameter_tuning": {
            "enabled": True,
            "method": "grid",
            "n_trials": 4,
            "search_optimizations": {
                "with_eda": False,
                "with_analysis": False,
                "interpret": False,
                "visualize": False,
                "model": {
                    "training": {
                        "epochs": 2,
                        "early_stopping_patience": 1,
                    }
                },
            },
            "model_hyperparams": {
                "CNN": {
                    "learning_rate": [0.001, 0.01],
                    "batch_size": [16, 32],
                    "cnn_variants": [
                        {"name": "tiny", "filters": [16], "kernel_sizes": [3]}
                    ],
                }
            },
        },
        "model": {
            "training": {"epochs": 100, "batch_size": 32, "validation_split": 0.2},
            "cnn": {
                "filters": [32],
                "kernel_sizes": [3],
                "strides": [1],
                "paddings": ["same"],
                "pool_sizes": [2],
                "pool_strides": [2],
            },
        },
    }


def test_hyperparameter_tuning_config_integration():
    """Test that hyperparameter tuning integrates properly with config system."""
    print("🔄 Testing config integration...")

    config = create_minimal_tuning_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        tuner = HyperparameterTuner(temp_config_path)

        # Test parameter combination generation
        combinations = tuner.generate_parameter_combinations()

        # Test config preparation for each combination
        for i, combo in enumerate(combinations):
            config_result, hyperparams = tuner.prepare_model_config(combo)

            # Verify that base configuration is preserved
            assert config_result["project"] == "fruitfly_aging"
            assert config_result["data"]["tissue"] == "head"

            # Verify that search optimizations are applied
            assert config_result["with_eda"] is False
            assert config_result["with_analysis"] is False
            assert (
                config_result["model"]["training"]["epochs"] == 2
            )  # Optimized for search

            # Verify that hyperparameters are properly extracted
            assert "learning_rate" in hyperparams
            assert "batch_size" in hyperparams
            assert hyperparams["learning_rate"] in [0.001, 0.01]
            assert hyperparams["batch_size"] in [16, 32]

            # Verify that CNN config overrides are applied
            assert config_result["model"]["cnn"]["filters"] == [16]  # From tiny variant

            # Verify that hyperparameter_tuning section is removed
            assert "hyperparameter_tuning" not in config_result

            print(f"✅ Config preparation validated for combination {i + 1}")

        print("✅ Config integration test PASSED!")
        return True

    finally:
        Path(temp_config_path).unlink()


@patch("timeflies.core.hyperparameter_tuner.OPTUNA_AVAILABLE", True)
def test_bayesian_optimization_setup():
    """Test that Bayesian optimization can be set up correctly."""
    print("🔄 Testing Bayesian optimization setup...")

    config = create_minimal_tuning_config()
    config["hyperparameter_tuning"]["method"] = "bayesian"
    config["hyperparameter_tuning"]["n_trials"] = 5

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        # Test with Optuna mocked as available
        tuner = HyperparameterTuner(temp_config_path)

        assert tuner.search_method == "bayesian"
        assert tuner.n_trials == 5

        # Test basic setup (without actual Optuna calls)
        tuner._setup_run_directory()

        # For testing, we just verify the tuner is configured correctly
        # without actually calling Optuna functions
        assert hasattr(tuner, "run_dir")
        assert tuner.run_dir is not None

        print("✅ Bayesian optimization setup successful")

        return True

    finally:
        Path(temp_config_path).unlink()


if __name__ == "__main__":
    print("🚀 Running hyperparameter tuning end-to-end tests...")
    test_hyperparameter_tuning_e2e_mock()
    test_hyperparameter_tuning_config_integration()
    test_bayesian_optimization_setup()
    print("🎉 All hyperparameter tuning e2e tests passed!")
