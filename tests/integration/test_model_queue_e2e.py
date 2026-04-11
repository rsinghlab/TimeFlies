"""End-to-end test for model queue functionality using tiny datasets."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml


def create_tiny_queue_config():
    """Create a minimal queue config for testing with tiny datasets."""
    return {
        "queue_settings": {
            "name": "tiny_e2e_test",
            "sequential": True,
            "save_checkpoints": True,
            "generate_summary": True,
        },
        "model_queue": [
            {
                "name": "logistic_tiny",
                "model_type": "logistic",
                "description": "Logistic regression on tiny dataset",
                "hyperparameters": {
                    "max_iter": 10,
                    "C": 1.0,
                },
            },
            {
                "name": "random_forest_tiny",
                "model_type": "random_forest",
                "description": "Random forest on tiny dataset",
                "hyperparameters": {
                    "n_estimators": 3,
                    "max_depth": 3,
                },
            },
        ],
        "global_settings": {
            "project": "fruitfly_aging",
            "data": {
                "tissue": "head",
                "target_variable": "age",
                "sampling": {
                    "samples": 20,
                    "variables": 25,
                },
                "batch_correction": {"enabled": False},
                "split": {
                    "method": "random",
                },
            },
            "with_training": True,
            "with_evaluation": True,
            "with_eda": False,
            "with_analysis": False,
            "interpret": False,
            "visualize": False,
        },
    }


def test_model_queue_execution_control():
    """Test model queue execution control options (train-only, eval-only, etc.)."""
    from timeflies.core.model_queue import ModelQueueManager

    print("🔄 Testing model queue execution control...")

    # Create config with mixed execution modes
    config = {
        "queue_settings": {
            "name": "execution_control_test",
            "sequential": True,
            "save_checkpoints": False,
            "generate_summary": False,
        },
        "model_queue": [
            {
                "name": "train_only_model",
                "model_type": "logistic",
                "description": "Train only test",
                "hyperparameters": {"max_iter": 5},
                "config_overrides": {
                    "with_training": True,
                    "with_evaluation": False,
                    "with_analysis": False,
                },
            },
            {
                "name": "eval_only_model",
                "model_type": "logistic",
                "description": "Evaluation only test",
                "hyperparameters": {"max_iter": 5},
                "config_overrides": {
                    "with_training": False,
                    "with_evaluation": True,
                    "with_analysis": False,
                },
            },
        ],
        "global_settings": {
            "project": "fruitfly_aging",
            "data": {"tissue": "head", "target_variable": "age"},
            "with_training": True,
            "with_evaluation": True,
            "with_analysis": False,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        with patch("timeflies.cli.commands.train_command") as mock_train:
            with patch("timeflies.cli.commands.evaluate_command") as mock_evaluate:
                mock_train.return_value = 0
                mock_evaluate.return_value = 0

                manager = ModelQueueManager(temp_config_path)

                # Test first model (train only)
                model1_config = manager.model_queue[0]
                config1 = manager.prepare_model_config(model1_config)

                assert config1.get("with_training") is True
                assert config1.get("with_evaluation") is False

                # Test second model (eval only)
                model2_config = manager.model_queue[1]
                config2 = manager.prepare_model_config(model2_config)

                assert config2.get("with_training") is False
                assert config2.get("with_evaluation") is True

                print("✅ Execution control configuration test PASSED!")

                return True

    finally:
        Path(temp_config_path).unlink()


if __name__ == "__main__":
    print("🚀 Running model queue end-to-end tests...")
    test_model_queue_e2e_with_tiny_dataset()
    test_model_queue_execution_control()
    print("🎉 All model queue e2e tests passed!")
