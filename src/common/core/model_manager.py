"""Model management utilities for pipeline operations."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model loading, building, training, and validation loss tracking."""

    def __init__(self, config, path_manager):
        self.config = config
        self.path_manager = path_manager

    def get_previous_best_loss_info(self, experiment_name: str) -> tuple[float | None, str]:
        """
        Get previous best validation loss and a descriptive message.

        Returns:
            Tuple of (best_loss_value, descriptive_message)
        """
        config_key = self.path_manager.get_config_key()
        base_path = Path(self.path_manager._get_project_root()) / "outputs"
        project_name = getattr(self.path_manager.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.path_manager.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        # Add task_type directory level to match current path structure
        task_type = getattr(self.path_manager.config.model, "task_type", "classification")

        best_symlink_path = str(
            base_path
            / project_name
            / "experiments"
            / correction_dir
            / task_type
            / "best"
            / config_key
            / "model_components"
            / "best_val_loss.json"
        )

        # Get current experiment components dir path
        experiment_dir = self.path_manager.get_experiment_dir(experiment_name)
        current_path = os.path.join(experiment_dir, "components", "best_val_loss.json")

        # Load the best validation loss from file
        best_val_loss = float("inf")
        model_found = False

        for path_to_try in [best_symlink_path, current_path]:
            try:
                if os.path.exists(path_to_try):
                    with open(path_to_try) as f:
                        best_val_loss = json.load(f)["best_val_loss"]
                        model_found = True
                        break
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        if model_found:
            message = f"Previous best validation loss: {best_val_loss:.3f}"
            return best_val_loss, message
        else:
            return None, "No previous model found"

    def build_model_with_architecture_display(self, model_builder, train_data):
        """Build model and display architecture information."""
        print("\n")
        print("MODEL ARCHITECTURE")
        print("=" * 60)
        model = model_builder.build_model()
        print("-" * 60)
        return model

    def display_model_architecture(self, model):
        """Display comprehensive model architecture information."""
        if model is None:
            return

        # Get model type
        print()
        model_type = getattr(self.config.data, "model", "CNN").lower()
        print(f"Model Type: {model_type.upper()}")

        # For TensorFlow/Keras models (CNN, MLP)
        if model_type in ["cnn", "mlp"]:
            self._display_keras_model_info(model)
        # For sklearn models (RF, LR, XGBoost)
        elif model_type in ["randomforest", "logisticregression", "xgboost"]:
            self._display_sklearn_model_info(model, model_type)

    def _display_keras_model_info(self, model):
        """Display training configuration and architecture for Keras models."""
        print("\nTraining Configuration:")
        try:
            # Get optimizer info
            optimizer_name = "Unknown"
            learning_rate = "Unknown"
            if hasattr(model, "optimizer"):
                optimizer = model.optimizer
                optimizer_name = optimizer.__class__.__name__
                if hasattr(optimizer, "learning_rate"):
                    learning_rate = round(float(optimizer.learning_rate), 3)

            print(f"  └─ Optimizer:              {optimizer_name}")
            print(f"  └─ Learning Rate:          {learning_rate}")

            # Get training config from config
            batch_size = getattr(self.config.model.training, "batch_size", 32)
            max_epochs = getattr(self.config.model.training, "epochs", 100)

            print(f"  └─ Batch Size:             {batch_size}")
            print(f"  └─ Max Epochs:             {max_epochs}")
            print(f"  └─ Validation Split:      {getattr(self.config.model.training, 'validation_split', 0.2)}")
            print(f"  └─ Early Stopping Patience: {getattr(self.config.model.training, 'early_stopping_patience', 10)}")
            print()

        except Exception as e:
            print(f"  └─ Could not display training config: {e}")

        # Display model architecture
        try:
            import io
            import sys

            # Capture model.summary() output
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            model.summary()
            sys.stdout = old_stdout
            summary_output = buffer.getvalue()

            # Print the summary with some formatting
            for line in summary_output.split("\n"):
                if line.strip():
                    print(f"  {line}")

        except Exception as e:
            print(f"  └─ Could not display detailed architecture: {e}")

    def _display_sklearn_model_info(self, model, model_type: str):
        """Display configuration for sklearn models."""
        print("\nModel Configuration:")
        if hasattr(model, "get_params"):
            params = model.get_params()
            for param_name, param_value in sorted(params.items()):
                if param_value is not None:
                    print(f"  └─ {param_name:<20}: {param_value}")

        # Additional details for specific models
        if model_type == "randomforest" and hasattr(model, "n_estimators"):
            print(f"  └─ Number of Trees:      {model.n_estimators}")
        elif model_type == "xgboost" and hasattr(model, "n_estimators"):
            print(f"  └─ Number of Boosters:   {model.n_estimators}")

    def print_training_results(self, history, model_improved: bool,
                             original_previous_best_loss: float | None,
                             experiment_name: str):
        """Print training results section."""
        print("\n")
        print("RESULTS")
        print("=" * 60)

        # Training subsection
        print("Training:")
        current_best_loss = None

        if history:
            val_losses = history.history.get("val_loss", [])
            if val_losses:
                best_epoch = val_losses.index(min(val_losses)) + 1
                current_best_loss = min(val_losses)
                print(f"  └─ Best Epoch:                {best_epoch}")
                print(f"  └─ Best Val Loss (This Run):  {current_best_loss:.4f}")

        # Show previous best and change using ORIGINAL previous best
        if original_previous_best_loss is not None and current_best_loss is not None:
            delta = original_previous_best_loss - current_best_loss
            print(f"  └─ Previous Best Val Loss:    {original_previous_best_loss:.4f}")
            if delta > 0:
                print(f"  └─ Improvement:               -{delta:.4f} (Better)")
            elif delta < 0:
                print(f"  └─ Change:                    +{abs(delta):.4f} (Worse)")
            else:
                print(f"  └─ No Change:                 {delta:.4f}")
        else:
            print("  └─ Previous Best:             Not available")

        # Show improvement status
        improvement_status = (
            "New best model found"
            if model_improved
            else "No improvement over existing model found"
        )

        print(f"  └─ Model Saved To:            {experiment_name}")
        print(f"  └─ Status:                    {improvement_status}")
