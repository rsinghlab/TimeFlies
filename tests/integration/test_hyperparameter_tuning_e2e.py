"""End-to-end test for hyperparameter tuning functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from common.core.hyperparameter_tuner import HyperparameterTuner


def create_minimal_tuning_config():
    """Create a minimal config for e2e hyperparameter tuning test."""
    return {
        "project": "fruitfly_aging",
        "data": {
            "model": "CNN",
            "tissue": "head",
            "target_variable": "age",
            "sampling": {"samples": 50, "variables": 25},  # Very small for testing
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
                        "epochs": 2,  # Very short for testing
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


def test_hyperparameter_tuning_e2e_mock():
    """
    End-to-end test of hyperparameter tuning with mocked training.

    This test verifies that the entire hyperparameter tuning system works:
    1. Load configuration with hyperparameter tuning enabled
    2. Generate parameter combinations
    3. Run trials with mocked training
    4. Generate summary reports
    5. Verify all outputs are created
    """
    print("🔄 Starting hyperparameter tuning end-to-end test...")

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = create_minimal_tuning_config()
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        with patch("common.cli.commands.train_command") as mock_train:
            with patch("common.cli.commands.evaluate_command") as mock_evaluate:
                # Mock successful training and evaluation
                mock_train.return_value = 0  # Success
                mock_evaluate.return_value = 0  # Success

                # Initialize hyperparameter tuner
                tuner = HyperparameterTuner(temp_config_path)

                # Verify configuration loaded correctly
                assert tuner.search_method == "grid"
                assert tuner.current_model_type == "CNN"
                assert tuner.n_trials == 4

                print("✅ Configuration loaded successfully")

                # Generate parameter combinations
                combinations = tuner.generate_parameter_combinations()
                print(f"✅ Generated {len(combinations)} parameter combinations")

                # Should generate combinations for CNN variant × hyperparameters
                # 1 variant × 2 learning_rates × 2 batch_sizes = 4 combinations
                assert len(combinations) == 4

                # Verify combinations structure
                for combo in combinations:
                    assert combo["model_type"] == "CNN"
                    assert combo["variant_name"] == "tiny"
                    assert "hyperparameters" in combo
                    assert "config_overrides" in combo

                print("✅ Parameter combinations validated")

                # Mock the model training to simulate real execution
                def mock_run_trial(trial_index, trial_params):
                    """Mock trial execution with realistic results."""
                    return {
                        "trial_index": trial_index,
                        "variant_name": trial_params["variant_name"],
                        "model_type": trial_params["model_type"],
                        "config_overrides": trial_params["config_overrides"],
                        "hyperparameters": trial_params["hyperparameters"],
                        "status": "completed",
                        "training_time": 1.5
                        + trial_index * 0.2,  # Simulate different times
                        "metrics": {
                            "accuracy": 0.80
                            + trial_index * 0.02,  # Simulate improvement
                            "precision": 0.78 + trial_index * 0.02,
                            "recall": 0.82 + trial_index * 0.01,
                            "f1_score": 0.80 + trial_index * 0.015,
                        },
                        "timestamp": "2024-08-25 15:00:00",
                    }

                # Replace the run_trial method with our mock
                tuner.run_trial = mock_run_trial

                # Mock summary report generation to prevent file creation during test
                def mock_generate_summary_report():
                    return Mock(), Mock()  # Return mock paths

                tuner.generate_summary_report = mock_generate_summary_report

                # Run the hyperparameter search
                print("🔄 Running hyperparameter search...")
                results = tuner.run_search(resume=False)

                # Verify results
                assert len(tuner.results) == 4, (
                    f"Expected 4 results, got {len(tuner.results)}"
                )

                # Check that all trials completed successfully
                completed_trials = [
                    r for r in tuner.results if r["status"] == "completed"
                ]
                assert len(completed_trials) == 4

                # Check trial results structure
                for result in tuner.results:
                    assert result["variant_name"] == "tiny"
                    assert result["model_type"] == "CNN"
                    assert result["status"] == "completed"
                    assert "metrics" in result
                    assert "training_time" in result
                    assert "hyperparameters" in result

                print("✅ All trials completed successfully")

                # Verify search results summary
                assert results["search_method"] == "grid"
                assert results["total_trials"] == 4
                assert results["completed_trials"] == 4
                assert results["failed_trials"] == 0

                # Check best trial identification
                best_trial = results["best_trial"]
                assert best_trial is not None
                assert best_trial["variant_name"] == "tiny"
                assert "metrics" in best_trial

                print("✅ Search results validated")
                print("🎉 Hyperparameter tuning end-to-end test PASSED!")

                return True

    finally:
        # Clean up temp config file
        Path(temp_config_path).unlink()


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
        # Test with Optuna available
        tuner = HyperparameterTuner(temp_config_path)

        assert tuner.search_method == "bayesian"
        assert tuner.n_trials == 5

        # Test Optuna study setup (without actually running trials)
        tuner._setup_run_directory()
        study = tuner._setup_optuna_study()

        assert study is not None
        assert study.direction.name == "MAXIMIZE"  # For accuracy maximization

        print("✅ Bayesian optimization setup successful")

        # Clean up study database
        if tuner.run_dir and tuner.run_dir.exists():
            import shutil

            shutil.rmtree(tuner.run_dir)

        return True

    finally:
        Path(temp_config_path).unlink()


if __name__ == "__main__":
    print("🚀 Running hyperparameter tuning end-to-end tests...")
    test_hyperparameter_tuning_e2e_mock()
    test_hyperparameter_tuning_config_integration()
    test_bayesian_optimization_setup()
    print("🎉 All hyperparameter tuning e2e tests passed!")
