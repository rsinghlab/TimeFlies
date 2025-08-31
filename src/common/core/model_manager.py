"""Model management utilities for pipeline operations."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from common.display.display_manager import DisplayManager
from common.models.model import ModelBuilder, ModelLoader

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model loading, building, training, and validation loss tracking."""

    def __init__(self, config, path_manager):
        self.config = config
        self.path_manager = path_manager
        self.display_manager = DisplayManager(config)

    def get_previous_best_loss_info(
        self, experiment_name: str
    ) -> tuple[float | None, str]:
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
        task_type = getattr(
            self.path_manager.config.model, "task_type", "classification"
        )

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
        self.display_manager.print_header("MODEL ARCHITECTURE")
        model = model_builder.build_model()
        self.display_manager.display_model_architecture(model, self.config)
        return model

    def print_training_results(
        self,
        history,
        model_improved: bool,
        original_previous_best_loss: float | None,
        experiment_name: str,
    ):
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

    def load_model_components(self, config_instance):
        """Load pre-trained model and all its components for evaluation."""
        model_loader = ModelLoader(config_instance)

        # Load model components
        components = model_loader.load_model_components()

        # Load the actual model
        model = model_loader.load_model()

        return model, components, model_loader

    def build_model(
        self,
        config_instance,
        train_data,
        train_labels,
        label_encoder,
        reference_data,
        scaler,
        is_scaler_fit,
        highly_variable_genes,
        mix_included,
        experiment_name,
    ):
        """Build the model without training it."""
        # Build the model
        model_builder = ModelBuilder(
            config_instance,
            train_data,
            train_labels,
            label_encoder,
            reference_data,
            scaler,
            is_scaler_fit,
            highly_variable_genes,
            mix_included,
            experiment_name,
        )

        # Build the model only (without training)
        model = model_builder.build_model()

        return model, model_builder

    def train_model(self, model_builder, model):
        """Train the pre-built model."""
        # Train the model using the provided training data and additional components
        history, trained_model, model_improved = model_builder.train_model(model)

        return history, trained_model, model_improved

    def get_previous_best_loss_message(self, experiment_name: str) -> str:
        """Get previous best validation loss message for header."""
        config_key = self.path_manager.get_config_key()
        # Get base path and manually construct the correct best path
        base_path = Path(self.path_manager._get_project_root()) / "outputs"
        project_name = getattr(self.path_manager.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.path_manager.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        task_type = getattr(
            self.path_manager.config.model, "task_type", "classification"
        )

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
            return f"Previous best validation loss: {best_val_loss:.3f}"
        else:
            return "No previous model found"

    def get_previous_best_validation_loss(self, experiment_name: str) -> float | None:
        """Get just the numerical value of previous best validation loss."""
        config_key = self.path_manager.get_config_key()
        base_path = Path(self.path_manager._get_project_root()) / "outputs"
        project_name = getattr(self.path_manager.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.path_manager.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        task_type = getattr(
            self.path_manager.config.model, "task_type", "classification"
        )

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

        experiment_dir = self.path_manager.get_experiment_dir(experiment_name)
        current_path = os.path.join(experiment_dir, "components", "best_val_loss.json")

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

        return best_val_loss if model_found else None
