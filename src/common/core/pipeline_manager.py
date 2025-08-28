import gc
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

# Suppress TensorFlow and gRPC logging
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common.data.loaders import DataLoader
from common.data.preprocessing.data_processor import DataPreprocessor
from common.data.preprocessing.gene_filter import GeneFilter
from common.models.model import ModelBuilder, ModelLoader
from common.utils.gpu_handler import GPUHandler
from common.utils.path_manager import PathManager
from common.utils.storage_manager import StorageManager

from .config_manager import Config

# Optional project-specific components (injected at runtime)
EDAHandler = None
Interpreter = None
Metrics = None
Visualizer = None

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    The PipelineManager class orchestrates the entire data processing, modeling,
    and visualization pipeline. It handles tasks such as GPU configuration, data loading,
    preprocessing, model training/loading, interpretation, and visualization based on the
    configurations provided.

    Attributes:
        config_instance (ConfigHandler): The configuration object containing all settings.
        data_loader (DataLoader): Instance responsible for loading data.
        path_manager (PathManager): Manages file paths based on the configuration.
    """

    def __init__(self, config: Config, mode: str = "training"):
        """
        Initialize the PipelineManager class with configuration and data loader.

        Args:
            config: Configuration object
            mode: "training" (creates new experiment) or "evaluation" (reuses best experiment)
        """
        self.config_instance = config
        self.data_loader = DataLoader(self.config_instance)
        self.path_manager = PathManager(self.config_instance)
        self.storage_manager = StorageManager(self.config_instance, self.path_manager)
        self.mode = mode

        # Set experiment name based on mode
        if mode == "training":
            # Generate new experiment for training
            self.experiment_name = self.path_manager.generate_experiment_name()
        else:
            # Reuse best experiment for standalone evaluation
            try:
                self.experiment_name = self.path_manager.get_best_experiment_name()
            except (FileNotFoundError, RuntimeError):
                # If no best experiment exists, this will fail later with a clear error
                self.experiment_name = None

        self.config_key = self.path_manager.get_config_key()
        # Config: {self.config_key}, Experiment: {self.experiment_name}

        # Auto-inject shared analysis components
        try:
            from common.analysis import EDAHandler, Visualizer
            from common.evaluation import Interpreter, Metrics

            self.eda_handler_class = EDAHandler
            self.interpreter_class = Interpreter
            self.metrics_class = Metrics
            self.visualizer_class = Visualizer
        except ImportError:
            self.eda_handler_class = None
            self.interpreter_class = None
            self.metrics_class = None
            self.visualizer_class = None

        # Initializing TimeFlies pipeline

    def _print_header(self, title: str, width: int = 60):
        """Print a formatted header with title."""
        print("\n" + "=" * width)
        print(title)
        print("=" * width)

    def _print_project_and_dataset_overview(self):
        """Print consolidated project information and actual training/evaluation data overview."""
        print("=" * 60)
        print("\n")
        print("DATA OVERVIEW")
        print("-" * 60)

        # Show training dataset size and distribution
        if hasattr(self, "adata") and self.adata is not None:
            train_cells = self.adata.n_obs
            train_genes = self.adata.n_vars
            print("Training Dataset Size:")
            print(f"  └─ Samples:           {train_cells:,}")
            print(f"  └─ Features (genes):  {train_genes:,}")

            # Show target distribution for training data
            target = getattr(self.config_instance.data, "target_variable", "age")
            target_name = target if isinstance(target, str) else str(target)
            if target in self.adata.obs.columns:
                print(f"\nTraining {target_name.title()} Distribution:")
                train_dist = self.adata.obs[target].value_counts().sort_index()
                train_total = train_dist.sum()
                for key, count in train_dist.items():
                    pct = (count / train_total) * 100
                    print(f"  └─ {key:<12}: {count:6,} samples ({pct:5.1f}%)")

        # Show evaluation dataset size and distribution
        if hasattr(self, "adata_eval") and self.adata_eval is not None:
            eval_cells = self.adata_eval.n_obs
            eval_genes = self.adata_eval.n_vars
            print("\nEvaluation Dataset Size:")
            print(f"  └─ Samples:           {eval_cells:,}")
            print(f"  └─ Features (genes):  {eval_genes:,}")

            # Show target distribution for evaluation data
            if target in self.adata_eval.obs.columns:
                print(f"\nEvaluation {target.title()} Distribution:")
                eval_dist = self.adata_eval.obs[target].value_counts().sort_index()
                eval_total = eval_dist.sum()
                for key, count in eval_dist.items():
                    pct = (count / eval_total) * 100
                    print(f"  └─ {key:<12}: {count:6,} samples ({pct:5.1f}%)")

        # Split configuration
        from common.utils.split_naming import SplitNamingUtils

        split_config = SplitNamingUtils.extract_split_details_for_metadata(
            self.config_instance
        )
        if split_config:
            print("\nSplit Configuration:")
            print(
                f"  └─ Split Method:      {split_config.get('method', 'unknown').title()}"
            )
            if split_config.get("method") == "column":
                print(
                    f"  └─ Split Column:      {split_config.get('column', 'unknown')}"
                )
                train_vals = split_config.get("train_values", [])
                test_vals = split_config.get("test_values", [])
                if train_vals:
                    print(f"  └─ Training Values:   {', '.join(train_vals)}")
                if test_vals:
                    print(f"  └─ Test Values:       {', '.join(test_vals)}")

        print("=" * 60)

    def _get_processed_eval_data_with_metadata(self):
        """
        Helper function that processes evaluation data exactly like the evaluation pipeline
        and returns both processed data and filtered metadata subset.

        Returns:
            tuple: (eval_data, eval_labels, label_encoder, eval_subset)
        """
        # Get evaluation data - use corrected if available, otherwise regular
        adata_eval = self.adata_eval_corrected or self.adata_eval

        if adata_eval is None:
            return None, None, None, None

        # Apply split filtering first to get the correct subset for metadata display
        split_method = getattr(self.config_instance.data.split, "method", "random").lower()
        filtered_adata_eval = adata_eval.copy()
        
        if split_method == "column":
            # Apply the same filtering logic as prepare_final_eval_data
            column = getattr(self.config_instance.data.split, "column", None)
            test_values = getattr(self.config_instance.data.split, "test", [])
            
            if column and test_values:
                # Convert values to lowercase strings for consistent matching
                test_values_norm = [str(v).lower() for v in test_values]
                
                # Filter to only test values for evaluation
                filtered_adata_eval = filtered_adata_eval[
                    filtered_adata_eval.obs[column].str.lower().isin(test_values_norm)
                ].copy()

        # Create eval_subset from the filtered data for accurate metadata display
        if hasattr(self, 'data_preprocessor') and self.data_preprocessor is not None:
            eval_subset = self.data_preprocessor.process_adata(filtered_adata_eval)
        else:
            # Fallback: create temp preprocessor
            temp_processor = DataPreprocessor(self.config_instance, filtered_adata_eval, None)
            eval_subset = temp_processor.process_adata(filtered_adata_eval)

        # Process evaluation data with fitted components from training (exactly like evaluation pipeline)
        if hasattr(self, 'data_preprocessor') and self.data_preprocessor is not None:
            eval_data, eval_labels, temp_label_encoder = self.data_preprocessor.prepare_final_eval_data(
                adata_eval,  # Use original adata_eval since prepare_final_eval_data does its own filtering
                getattr(self, "label_encoder", None),
                getattr(self, "num_features", None),
                getattr(self, "scaler", None),
                getattr(self, "is_scaler_fit", False),
                getattr(self, "highly_variable_genes", None),
                getattr(self, "mix_included", False),
                getattr(self, "train_gene_names", None),
            )
        else:
            # Fallback: create temp preprocessor
            temp_processor = DataPreprocessor(self.config_instance, adata_eval, None)
            eval_data, eval_labels, temp_label_encoder = temp_processor.prepare_final_eval_data(
                adata_eval, None, None, None, False, None, False, None
            )

        return eval_data, eval_labels, temp_label_encoder, eval_subset

    def _print_obs_distributions(self, adata, columns):
        """Helper to print distributions for specified obs columns."""
        for col in columns:
            if col in adata.obs.columns:
                print(f"  └─ {col.replace('_', ' ').title()} Distribution:")
                counts = adata.obs[col].value_counts().sort_index()
                for value, count in counts.items():
                    percentage = (count / len(adata.obs)) * 100
                    print(f"      └─ {value:<12}: {count:6,} samples ({percentage:5.1f}%)")

    def _print_training_and_evaluation_data(self):
        """Print training and evaluation data details after preprocessing."""
        print("\n")
        print("PREPROCESSED DATA OVERVIEW")
        print("=" * 60)

        # Check if we have any preprocessed data to display
        if not (hasattr(self, "train_data") or hasattr(self, "test_data")):
            print("No preprocessed data available to display.")
            return

        import numpy as np

        # Training data details
        if hasattr(self, "train_data") and self.train_data is not None:
            try:
                train_shape = self.train_data.shape
                print("Training Data:")
                print(f"  └─ Samples:           {train_shape[0]:,}")
                if len(train_shape) > 2:  # CNN format
                    print(
                        f"  └─ Features:          {train_shape[1]} x {train_shape[2]:,} (reshaped)"
                    )
                else:  # Standard format
                    print(f"  └─ Features (genes):  {train_shape[1]:,}")

                # Data statistics
                train_mean = np.mean(self.train_data)
                train_std = np.std(self.train_data)
                train_min = np.min(self.train_data)
                train_max = np.max(self.train_data)
                print(f"  └─ Data Range:        [{train_min:.3f}, {train_max:.3f}]")
                print(f"  └─ Mean ± Std:        {train_mean:.3f} ± {train_std:.3f}")

                # Training labels distribution - integrated into training data section
                if hasattr(self, "train_labels") and self.train_labels is not None:
                    encoding_var = getattr(
                        self.config_instance.data, "target_variable", "target"
                    )
                    print(f"  └─ {encoding_var.title()} Distribution:")

                    try:
                        # Handle both one-hot encoded and label encoded data
                        if (
                            hasattr(self.train_labels, "shape")
                            and len(self.train_labels.shape) > 1
                            and self.train_labels.shape[1] > 1
                        ):
                            # One-hot encoded - convert to class indices
                            label_indices = np.argmax(self.train_labels, axis=1)
                        else:
                            # Already class indices or 1D array
                            label_indices = np.array(self.train_labels).flatten()

                        unique, counts = np.unique(label_indices, return_counts=True)
                        total = counts.sum()
                        for label_encoded, count in zip(unique, counts):
                            pct = (count / total) * 100
                            try:
                                if (
                                    hasattr(self, "label_encoder")
                                    and self.label_encoder is not None
                                ):
                                    original_label = self.label_encoder.inverse_transform(
                                        [int(label_encoded)]
                                    )[0]
                                    print(
                                        f"      └─ {original_label:<12}: {count:6,} samples ({pct:5.1f}%)"
                                    )
                                else:
                                    print(
                                        f"      └─ {label_encoded:<12}: {count:6,} samples ({pct:5.1f}%)"
                                    )
                            except Exception:
                                # Fallback to showing encoded label
                                print(
                                    f"      └─ Class {label_encoded:<7}: {count:6,} samples ({pct:5.1f}%)"
                                )
                    except Exception as e:
                        print(f"      └─ Could not display label distribution: {e}")

                # Additional distributions using helper
                if hasattr(self, 'train_subset') and self.train_subset is not None:
                    # Get display columns from config - only show what's explicitly listed
                    display_columns = list(getattr(self.config_instance.data, "display_columns", ["sex", "genotype"]))
                    self._print_obs_distributions(self.train_subset, display_columns)

            except Exception as e:
                print(f"Training Data: Could not display details ({e})")
        else:
            print("Training Data: Not available")

        # Always show processed evaluation data from separate eval dataset (final holdout)
        self._show_processed_eval_data_preview()

        # Check split method to decide whether to also show validation data
        split_method = getattr(self.config_instance.data.split, "method", "unknown")

        # For non-column splitting, also show validation data from train/test split
        if split_method != "column":
            if hasattr(self, "test_data") and self.test_data is not None:
                try:
                    test_shape = self.test_data.shape
                    if test_shape[0] > 0:  # Only show if we have samples
                        print("\nValidation Data:")
                        print(f"  └─ Samples:           {test_shape[0]:,}")
                        if len(test_shape) > 2:  # CNN format
                            print(
                                f"  └─ Features:          {test_shape[1]} x {test_shape[2]:,} (reshaped)"
                            )
                        else:  # Standard format
                            print(f"  └─ Features (genes):  {test_shape[1]:,}")

                        # Data statistics
                        test_mean = np.mean(self.test_data)
                        test_std = np.std(self.test_data)
                        test_min = np.min(self.test_data)
                        test_max = np.max(self.test_data)
                        print(f"  └─ Data Range:        [{test_min:.3f}, {test_max:.3f}]")
                        print(f"  └─ Mean ± Std:        {test_mean:.3f} ± {test_std:.3f}")

                        # Test labels distribution
                        if hasattr(self, "test_labels") and self.test_labels is not None:
                            encoding_var = getattr(
                                self.config_instance.data, "target_variable", "target"
                            )
                            print(f"\nValidation {encoding_var.title()} Distribution:")

                            # Handle both one-hot encoded and label encoded data
                            try:
                                if (
                                    hasattr(self.test_labels, "shape")
                                    and len(self.test_labels.shape) > 1
                                    and self.test_labels.shape[1] > 1
                                ):
                                    # One-hot encoded - convert to class indices
                                    label_indices = np.argmax(self.test_labels, axis=1)
                                else:
                                    # Already class indices or 1D array
                                    label_indices = np.array(self.test_labels).flatten()

                                unique, counts = np.unique(
                                    label_indices, return_counts=True
                                )
                                total = counts.sum()
                                for label_encoded, count in zip(unique, counts):
                                    pct = (count / total) * 100
                                    try:
                                        if (
                                            hasattr(self, "label_encoder")
                                            and self.label_encoder is not None
                                        ):
                                            original_label = (
                                                self.label_encoder.inverse_transform(
                                                    [int(label_encoded)]
                                                )[0]
                                            )
                                            print(
                                                f"  └─ {original_label:<12}: {count:6,} samples ({pct:.1f}%)"
                                            )
                                        else:
                                            print(
                                                f"  └─ {label_encoded:<12}: {count:6,} samples ({pct:.1f}%)"
                                            )
                                    except Exception:
                                        # Fallback to showing encoded label
                                        print(
                                            f"  └─ Class {label_encoded:<7}: {count:6,} samples ({pct:.1f}%)"
                                        )
                            except Exception as e:
                                print(f"  └─ Could not display label distribution: {e}")
                except Exception as e:
                    print(f"\nValidation Data: Could not display details ({e})")
            else:
                print("\nValidation Data: Not available")

        # Split configuration
        from common.utils.split_naming import SplitNamingUtils

        split_config = SplitNamingUtils.extract_split_details_for_metadata(
            self.config_instance
        )
        if split_config:
            print("\nSplit Configuration:")
            print(
                f"  └─ Split Method:      {split_config.get('method', 'unknown').title()}"
            )
            if split_config.get("method") == "column":
                print(
                    f"  └─ Split Column:      {split_config.get('column', 'unknown')}"
                )
                train_vals = split_config.get("train_values", [])
                test_vals = split_config.get("test_values", [])
                if train_vals:
                    print(f"  └─ Training Values:   {', '.join(train_vals)}")
                if test_vals:
                    print(f"  └─ Test Values:       {', '.join(test_vals)}")

    def _display_model_architecture(self):
        """Display comprehensive model architecture information."""
        if not hasattr(self, "model") or self.model is None:
            return

        # Get model type
        print()
        model_type = getattr(self.config_instance.data, "model", "CNN").lower()
        print(f"Model Type: {model_type.upper()}")

        # For TensorFlow/Keras models (CNN, MLP)
        if model_type in ["cnn", "mlp"]:
            # Display training configuration first
            print("\nTraining Configuration:")
            try:
                # Get optimizer info
                optimizer_name = "Unknown"
                learning_rate = "Unknown"
                if hasattr(self.model, "optimizer"):
                    optimizer = self.model.optimizer
                    optimizer_name = optimizer.__class__.__name__
                    if hasattr(optimizer, "learning_rate"):
                        learning_rate = round(float(optimizer.learning_rate), 3)

                print(f"  └─ Optimizer:              {optimizer_name}")
                print(f"  └─ Learning Rate:          {learning_rate}")

                # Get training config from config
                batch_size = getattr(
                    self.config_instance.model.training, "batch_size", 32
                )
                max_epochs = getattr(self.config_instance.model.training, "epochs", 100)

                print(f"  └─ Batch Size:             {batch_size}")
                print(f"  └─ Max Epochs:             {max_epochs}")
                print(
                    f"  └─ Validation Split:      {getattr(self.config_instance.model.training, 'validation_split', 0.2)}"
                )
                print(
                    f"  └─ Early Stopping Patience: {getattr(self.config_instance.model.training, 'early_stopping_patience', 10)}"
                )
                print()

            except Exception as e:
                print(f"  └─ Could not display training config: {e}")

            try:
                import io
                import sys

                # Capture model.summary() output
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()

                self.model.summary()

                sys.stdout = old_stdout
                summary_output = buffer.getvalue()

                # Print the summary with some formatting
                for line in summary_output.split("\n"):
                    if line.strip():
                        print(f"  {line}")

            except Exception as e:
                print(f"  └─ Could not display detailed architecture: {e}")

        # For sklearn models (RF, LR, XGBoost)
        elif model_type in ["randomforest", "logisticregression", "xgboost"]:
            print("\nModel Configuration:")
            if hasattr(self.model, "get_params"):
                params = self.model.get_params()
                for param_name, param_value in sorted(params.items()):
                    if param_value is not None:
                        print(f"  └─ {param_name:<20}: {param_value}")

            # Additional details for specific models
            if model_type == "randomforest" and hasattr(self.model, "n_estimators"):
                print(f"  └─ Number of Trees:      {self.model.n_estimators}")
            elif model_type == "xgboost" and hasattr(self.model, "n_estimators"):
                print(f"  └─ Number of Boosters:   {self.model.n_estimators}")

    def _show_processed_eval_data_preview(self):
        """Show preview of actual processed evaluation data."""
        try:
            print("\nEvaluation Data:")

            # Use helper function to get processed eval data exactly like evaluation pipeline
            eval_data, eval_labels, temp_label_encoder, eval_subset = self._get_processed_eval_data_with_metadata()

            if eval_data is None:
                print("  └─ Could not process evaluation data")
                return

            # Display the processed evaluation data
            eval_shape = eval_data.shape
            print(f"  └─ Samples:           {eval_shape[0]:,}")
            if len(eval_shape) > 2:  # CNN format
                print(
                    f"  └─ Features:          {eval_shape[1]} x {eval_shape[2]:,} (reshaped)"
                )
            else:  # Standard format
                print(f"  └─ Features (genes):  {eval_shape[1]:,}")

            # Data statistics
            import numpy as np

            eval_mean = np.mean(eval_data)
            eval_std = np.std(eval_data)
            eval_min = np.min(eval_data)
            eval_max = np.max(eval_data)
            print(f"  └─ Data Range:        [{eval_min:.3f}, {eval_max:.3f}]")
            print(f"  └─ Mean ± Std:        {eval_mean:.3f} ± {eval_std:.3f}")

            # Label distribution
            if eval_labels is not None and temp_label_encoder is not None:
                target = getattr(
                    self.config_instance.data, "target_variable", "age"
                )
                target_name = target if isinstance(target, str) else str(target)
                print(f"  └─ {target_name.title()} Distribution:")

                # Handle both one-hot encoded and label encoded data
                if len(eval_labels.shape) > 1 and eval_labels.shape[1] > 1:
                    # One-hot encoded
                    label_indices = np.argmax(eval_labels, axis=1)
                else:
                    # Already class indices
                    label_indices = np.array(eval_labels).flatten()

                unique, counts = np.unique(label_indices, return_counts=True)
                total = counts.sum()
                for label_encoded, count in zip(unique, counts):
                    pct = (count / total) * 100
                    try:
                        original_label = temp_label_encoder.inverse_transform(
                            [int(label_encoded)]
                        )[0]
                        print(
                            f"      └─ {original_label:<12}: {count:6,} samples ({pct:5.1f}%)"
                        )
                    except Exception:
                        print(
                            f"      └─ Class {label_encoded:<7}: {count:6,} samples ({pct:5.1f}%)"
                        )

            # Display metadata distributions from eval_subset
            if eval_subset is not None:
                # Get display columns from config - only show what's explicitly listed
                display_columns = list(getattr(self.config_instance.data, "display_columns", ["sex", "genotype"]))
                self._print_obs_distributions(eval_subset, display_columns)

        except Exception as e:
            print(f"\nEvaluation Data: Could not process preview ({e})")
            # Try to show raw data info if available
            try:
                adata_eval = getattr(self, "adata_eval_corrected", None) or getattr(
                    self, "adata_eval", None
                )
                if adata_eval is not None:
                    eval_cells = adata_eval.n_obs
                    eval_genes = adata_eval.n_vars
                    print(f"  └─ Raw Samples:       {eval_cells:,}")
                    print(f"  └─ Raw Features:      {eval_genes:,}")
                else:
                    print("  └─ No evaluation data available")
            except Exception as fallback_e:
                print(f"  └─ Could not show fallback info: {fallback_e}")

    def _get_previous_best_loss_message(self):
        """Get previous best validation loss message for header."""
        import json
        import os
        from pathlib import Path

        # Duplicate exact logic from model file that was working
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
        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
        current_path = os.path.join(experiment_dir, "components", "best_val_loss.json")

        # Load the best validation loss from file - exact duplicate from model file
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

    def _get_previous_best_validation_loss(self):
        """Get just the numerical value of previous best validation loss."""
        import json
        import os
        from pathlib import Path

        # Use same logic as _get_previous_best_loss_message but return just the value
        config_key = self.path_manager.get_config_key()
        base_path = Path(self.path_manager._get_project_root()) / "outputs"
        project_name = getattr(self.path_manager.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.path_manager.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

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



        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
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

    def _print_final_timing_summary(self, evaluation_duration):
        """Print timing information with preprocessing, training, evaluation, and total time."""
        print()
        print("TIMING")

        # Get timing data from stored values
        preprocessing_duration = getattr(self, "_preprocessing_duration", 0)
        training_duration = getattr(self, "_stored_training_duration", 0)

        # Calculate actual training time (excluding preprocessing)
        actual_training_duration = training_duration - preprocessing_duration
        total_time = training_duration + evaluation_duration

        print(f"  └─ Preprocessing Duration:    {preprocessing_duration:.1f} seconds")
        print(f"  └─ Model Training Duration:   {actual_training_duration:.1f} seconds")
        print(f"  └─ Evaluation Duration:       {evaluation_duration:.1f} seconds")
        print(f"  └─ Total Time:                {total_time:.1f} seconds")

    def _setup_pipeline(self):
        """Common pipeline setup: GPU, data loading, gene filtering."""
        # Setting up GPU
        GPUHandler.configure(self.config_instance)

        # Loading data
        (
            self.adata,
            self.adata_eval,
            self.adata_original,
        ) = self.data_loader.load_data()

        # Only load batch-corrected data if batch correction is enabled
        if self.config_instance.data.batch_correction.enabled:
            # Loading batch corrected data
            (
                self.adata_corrected,
                self.adata_eval_corrected,
            ) = self.data_loader.load_corrected_data()
        else:
            # Batch correction disabled
            self.adata_corrected = None
            self.adata_eval_corrected = None
        self.autosomal_genes, self.sex_genes = self.data_loader.load_gene_lists()

        # Applying gene filtering
        self.gene_filter = GeneFilter(
            self.config_instance,
            self.adata,
            self.adata_eval,
            self.adata_original,
            self.autosomal_genes,
            self.sex_genes,
        )
        (
            self.adata,
            self.adata_eval,
            self.adata_original,
        ) = self.gene_filter.apply_filter()

    def _cleanup_memory(self, keep_eval_data=True, keep_vis_data=False):
        """Clean up memory by deleting large data objects."""
        # Clean up training data, but keep visualization data if requested
        if not keep_vis_data:
            if hasattr(self, "adata"):
                del self.adata
            if hasattr(self, "adata_corrected") and self.adata_corrected is not None:
                del self.adata_corrected

        # Always clean up original data (not needed after preprocessing)
        if hasattr(self, "adata_original"):
            del self.adata_original

        # Optionally clean up evaluation data
        if not keep_eval_data:
            if hasattr(self, "adata_eval"):
                del self.adata_eval
            if (
                hasattr(self, "adata_eval_corrected")
                and self.adata_eval_corrected is not None
            ):
                del self.adata_eval_corrected

        gc.collect()

    def run_eda(self):
        """
        Perform Exploratory Data Analysis (EDA) if specified in the config.
        """
        if self.config_instance.data_processing.exploratory_data_analysis.enabled:
            # Running EDA
            if self.eda_handler_class is not None:
                eda_handler = self.eda_handler_class(
                    self.config_instance,
                    self.path_manager,
                    self.adata,
                    self.adata_eval,
                    self.adata_original,
                    self.adata_corrected,
                    self.adata_eval_corrected,
                )
                eda_handler.run_eda()
                # EDA completed
            else:
                logger.warning("EDA requested but no EDA handler available")
        else:
            # EDA disabled
            pass

    def preprocess_data(self, for_evaluation=False):
        """
        Preprocess data for training or evaluation.

        Args:
            for_evaluation: True for evaluation data (uses fitted components),
                          False for training data (fits new components)
        """
        self.data_preprocessor = DataPreprocessor(
            self.config_instance, self.adata, self.adata_corrected
        )

        if for_evaluation:
            # Skip if we already processed eval data in combined pipeline
            if hasattr(self, 'eval_data') and self.eval_data is not None:
                # Already processed during training phase
                self.test_data = self.eval_data
                self.test_labels = self.eval_labels
                return

            # Evaluation: use fitted components from training to prevent data leakage
            adata_eval = self.adata_eval_corrected or self.adata_eval
            (
                self.test_data,
                self.test_labels,
                self.label_encoder,
            ) = self.data_preprocessor.prepare_final_eval_data(
                adata_eval,
                self.label_encoder,
                self.num_features,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.mix_included,
                getattr(self, "train_gene_names", None),
            )
            # Clean up all memory (evaluation is final step)
            self._cleanup_memory(keep_eval_data=False)
        else:
            # Training: fit new components and prepare training/test splits
            (
                self.train_data,
                self.test_data,
                self.train_labels,
                self.test_labels,
                self.label_encoder,
                self.reference_data,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.mix_included,
                self.train_subset,
                self.test_subset,
            ) = self.data_preprocessor.prepare_data()

            # If combined pipeline, also process eval data NOW with fitted components
            is_combined_pipeline = getattr(self, "_in_combined_pipeline", False)
            if is_combined_pipeline:
                # Process evaluation data with fitted components for display
                adata_eval = self.adata_eval_corrected or self.adata_eval
                if adata_eval is not None:
                    (
                        self.eval_data,
                        self.eval_labels,
                        _,  # We already have label encoder
                    ) = self.data_preprocessor.prepare_final_eval_data(
                        adata_eval,
                        self.label_encoder,
                        self.train_data.shape[1],  # num_features from training
                        self.scaler,
                        self.is_scaler_fit,
                        self.highly_variable_genes,
                        self.mix_included,
                        getattr(self.data_preprocessor, "train_gene_names", None),
                    )
                else:
                    self.eval_data = None
                    self.eval_labels = None

            # Clean up memory (preserve evaluation data for auto-evaluation)
            self._cleanup_memory(
                keep_eval_data=True, keep_vis_data=is_combined_pipeline
            )

    def load_model(self):
        """
        Load pre-trained model and all its components for evaluation.
        """
        self.model_loader = ModelLoader(self.config_instance)

        # Load model components
        (
            self.label_encoder,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.num_features,
            self.history,
            self.mix_included,
            self.reference_data,
        ) = self.model_loader.load_model_components()

        # Load the actual model
        self.model = self.model_loader.load_model()

    def build_model(self):
        """
        Build the model without training it.
        """
        # Build the model
        self.model_builder = ModelBuilder(
            self.config_instance,
            self.train_data,
            self.train_labels,
            self.label_encoder,
            self.reference_data,
            self.scaler,
            self.is_scaler_fit,
            self.highly_variable_genes,
            self.mix_included,
            self.experiment_name,
        )

        # Build the model only (without training)
        print("\n")
        print("MODEL ARCHITECTURE")
        print("=" * 60)
        self.model = self.model_builder.build_model()
        print("-" * 60)

    def train_model(self):
        """
        Train the pre-built model.
        """
        # Train the model using the provided training data and additional components
        self.history, self.model, self.model_improved = self.model_builder.train_model(
            self.model
        )

        # Set num_features and gene names for auto-evaluation
        self.num_features = self.train_data.shape[1]
        # Store the exact gene names used in training for consistent evaluation
        if hasattr(self.data_preprocessor, "train_gene_names"):
            self.train_gene_names = self.data_preprocessor.train_gene_names
        else:
            self.train_gene_names = None

        # Save training visuals to experiment directory
        if self.config_instance.visualizations.enabled:
            # Generating training visualizations
            self.save_outputs(training_visuals=True)  # Save to experiment directory

    def save_outputs(
        self,
        training_visuals=False,
        evaluation_visuals=False,
        metadata=False,
        symlinks=False,
        result_type=None,
        ):
        """
        Unified method to save various pipeline outputs.

        Args:
            training_visuals: Save training visualizations (loss/accuracy curves)
            evaluation_visuals: Save evaluation visualizations
            metadata: Save experiment metadata
            symlinks: Update symlinks (latest/best)
            result_type: "recent" or "best" for symlink directories, None for experiment directory
        """
        # Save metadata if requested
        if metadata:
            training_data = {
                "training": {
                    "epochs_run": len(self.history.history.get("loss", [])),
                    "best_epoch": getattr(self, "best_epoch", None),
                    "best_val_loss": min(
                        self.history.history.get("val_loss", [float("inf")])
                    ),
                    "model_improved": getattr(self, "model_improved", False),
                },
            }

            # Only add evaluation data if it exists
            if hasattr(self, "last_mae"):
                training_data["evaluation"] = {
                    "mae": self.last_mae,
                    "r2": getattr(self, "last_r2", None),
                    "pearson": getattr(self, "last_pearson", None),
                }

            self.path_manager.save_experiment_metadata(
                self.experiment_name, training_data
            )

        # Save model and update folders if requested
        if symlinks:
            # For training+evaluation pipeline, always save the model
            # For standalone, respect the storage policy
            model_path = self.path_manager.get_experiment_model_path(
                self.experiment_name
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)

            # Update best folder if this is a new best model
            if self.model_improved:
                self.path_manager.update_best_folder(self.experiment_name)

            # Always update latest folder
            self.path_manager.update_latest_folder(self.experiment_name)

        # Save visualizations if requested and available
        if (
            training_visuals or evaluation_visuals
        ) and self.visualizer_class is not None:
            # Determine what data to pass based on visualization type
            test_data = self.test_data if evaluation_visuals else None
            test_labels = self.test_labels if evaluation_visuals else None
            shap_values = (
                getattr(self, "squeezed_shap_values", None)
                if evaluation_visuals
                else None
            )
            shap_data = (
                getattr(self, "squeezed_test_data", None)
                if evaluation_visuals
                else None
            )
            adata = getattr(self, "adata", None) if evaluation_visuals else None
            adata_corrected = (
                getattr(self, "adata_corrected", None) if evaluation_visuals else None
            )

            # Create visualizer
            visualizer = self.visualizer_class(
                self.config_instance,
                self.model,
                self.history,
                test_data,
                test_labels,
                self.label_encoder,
                shap_values,
                shap_data,
                adata,
                adata_corrected,
                self.path_manager,
            )

            # Set evaluation context if result_type provided (for symlinks)
            if result_type:
                visualizer.set_evaluation_context(result_type)
            elif evaluation_visuals or training_visuals:
                visualizer.set_evaluation_context(self.experiment_name)

            # Generate appropriate visualizations
            if training_visuals:
                visualizer._visualize_training_history()

            if evaluation_visuals:
                # Set output directory for evaluation plots
                plots_dir = self.path_manager.get_experiment_plots_dir(
                    self.experiment_name
                )
                visualizer.visual_tools.output_dir = plots_dir
                visualizer.run()

        elif (training_visuals or evaluation_visuals) and self.visualizer_class is None:
            logger.warning("Visualizer class not available, skipping visualizations")

    def run_training(self, skip_setup=False) -> dict[str, Any]:
        """
        Run the complete training pipeline: setup + preprocessing + model training.

        Args:
            skip_setup: Skip pipeline setup if already done (for combined pipeline)

        Returns:
            Dict containing training results and paths
        """
        import time

        if not skip_setup:
            self._print_header("🔥 TRAINING PIPELINE")
            self._setup_pipeline()

        # Start training timer to include preprocessing
        self._training_start_time = time.time()

        # Preprocessing with timing
        preprocessing_start_time = time.time()
        self.preprocess_data()
        self._preprocessing_duration = time.time() - preprocessing_start_time

        # Display training and evaluation data
        self._print_training_and_evaluation_data()

        # Build model first
        self.build_model()

        # Display model architecture after building but before training
        self._display_model_architecture()

        # Model training
        print("\n")
        print(f"TRAINING PROGRESS ({self._get_previous_best_loss_message()})")
        print("=" * 60)

        # Store original previous best loss for comparison (before training updates it)
        self._original_previous_best_loss = self._get_previous_best_validation_loss()

        self.train_model()

        # Save outputs and create symlinks after training
        self.save_outputs(metadata=True, symlinks=True)

        # Print training completion summary with improvement status
        improvement_status = (
            "New best model found"
            if self.model_improved
            else "No improvement over existing model found"
        )

        # Results section
        print("\n")
        print("RESULTS")
        print("=" * 60)

        # Training subsection
        print("Training:")
        current_best_loss = None
        original_previous_best_loss = getattr(self, "_original_previous_best_loss", None)

        if hasattr(self, "history") and self.history:
            val_losses = self.history.history.get("val_loss", [])
            if val_losses:
                best_epoch = val_losses.index(min(val_losses)) + 1
                current_best_loss = min(val_losses)
                print(f"  └─ Best Epoch:                {best_epoch}")
                print(f"  └─ Best Val Loss (This Run):  {current_best_loss:.4f}")

        # Show previous best and change using ORIGINAL previous best (not updated one)
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

        # Show training duration in Training section
        training_duration = 0
        if hasattr(self, "_training_start_time"):
            import time

            training_duration = time.time() - self._training_start_time
            # Store for later use in final summary
            self._stored_training_duration = training_duration

        print(f"  └─ Model Saved To:            {self.experiment_name}")
        print(f"  └─ Status:                    {improvement_status}")

        # Return training results
        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
        return {
            "model_path": experiment_dir,
            "experiment_name": self.experiment_name,
        }

    def run_interpretation(self):
        """
        Perform SHAP interpretation for model predictions.
        """
        if self.config_instance.interpretation.shap.enabled:
            try:
                logger.info("Starting SHAP interpretation...")
                if self.interpreter_class is None:
                    logger.warning(
                        "Interpreter class not available, skipping SHAP interpretation"
                    )
                    return

                # Initialize the Interpreter with the best model and data
                self.interpreter = self.interpreter_class(
                    self.config_instance,
                    self.model,
                    self.test_data,
                    self.test_labels,
                    self.label_encoder,
                    self.reference_data,
                    self.path_manager,
                )

                # Compute or load SHAP values
                (
                    squeezed_shap_values,
                    squeezed_test_data,
                ) = self.interpreter.compute_or_load_shap_values()

                # Update SHAP-related attributes for visualization
                self.squeezed_shap_values = squeezed_shap_values
                self.squeezed_test_data = squeezed_test_data

                logger.info("SHAP interpretation complete.")

            except Exception as e:
                logger.error(f"Error during SHAP interpretation: {e}")
                raise
        else:
            self.squeezed_shap_values, self.squeezed_test_data = None, None
            logger.info(
                "Skipping SHAP interpretation as it is disabled in the configuration."
            )

    def run_visualizations(self):
        """
        Set up and run visualizations for the current experiment.
        """
        if getattr(self.config_instance.visualizations, "enabled", False):
            # Running visualizations

            # Ensure SHAP attributes exist (set to None if not available)
            if not hasattr(self, "squeezed_shap_values"):
                self.squeezed_shap_values = None
            if not hasattr(self, "squeezed_test_data"):
                self.squeezed_test_data = None

            # Ensure adata attributes exist (may be deleted for memory cleanup)
            adata = getattr(self, "adata", None)
            adata_corrected = getattr(self, "adata_corrected", None)

            if self.visualizer_class is None:
                logger.warning(
                    "Visualizer class not available, skipping visualizations"
                )
                return

            visualizer = self.visualizer_class(
                self.config_instance,
                self.model,
                self.history,
                self.test_data,
                self.test_labels,
                self.label_encoder,
                self.squeezed_shap_values,
                self.squeezed_test_data,
                adata,
                adata_corrected,
                self.path_manager,
                preserved_var_names=None,  # Gene names preserved in model files
            )

            # Set the evaluation context for proper directory structure
            visualizer.set_evaluation_context(self.experiment_name)

            visualizer.run()
            # Visualizations completed
        else:
            logger.info("Visualization is disabled in the configuration.")

    def run_metrics(self, result_type="recent"):
        """
        Compute and display evaluation metrics for the model.

        Args:
            result_type: "recent" (standalone evaluation) or "best" (post-training)
        """
        # Computing evaluation metrics
        if self.metrics_class is None:
            logger.warning("Metrics class not available, skipping metrics computation")
            return

        # Get experiment evaluation directory for this result type
        evaluation_dir = self.path_manager.get_experiment_evaluation_dir(
            self.experiment_name
        )

        metrics = self.metrics_class(
            self.config_instance,
            self.model,
            self.test_data,
            self.test_labels,
            self.label_encoder,
            self.path_manager,
            result_type,
            output_dir=evaluation_dir,
        )
        metrics.compute_metrics()

    def run_analysis_script(self):
        """
        Run project-specific analysis script if available.
        """
        try:
            # Check for custom analysis script first
            if hasattr(self.config_instance, "_custom_analysis_script"):
                self._run_custom_analysis_script(
                    self.config_instance._custom_analysis_script
                )
                return

            logger.debug("Running project-specific analysis script...")

            # Auto-detect analysis script in templates folder
            project_name = self.config_instance.project

            # Look for templates/{project}_analysis.py
            from pathlib import Path

            template_path = Path("templates") / f"{project_name}_analysis.py"

            if template_path.exists():
                logger.debug(f"Found project analysis script: {template_path}")
                self._run_custom_analysis_script(str(template_path))
            else:
                logger.info(f"No analysis script found at {template_path}")
                logger.info("Available templates:")
                templates_dir = Path("templates")
                if templates_dir.exists():
                    template_files = [
                        f.name for f in templates_dir.glob("*_analysis.py")
                    ]
                    if template_files:
                        for template_file in template_files:
                            logger.info(f"  - {template_file}")
                    else:
                        logger.info("  - No analysis templates found")

        except Exception as e:
            logger.error(f"Error running analysis script: {e}")
            # Don't raise - analysis scripts are optional

    def _run_custom_analysis_script(self, script_path: str):
        """
        Run a custom analysis script provided by the user.

        Args:
            script_path: Path to the custom analysis Python file
        """
        try:
            script_path = Path(script_path)
            if not script_path.exists():
                logger.error(f"Custom analysis script not found: {script_path}")
                return

            if not script_path.suffix == ".py":
                logger.error(
                    f"Custom analysis script must be a .py file: {script_path}"
                )
                return

            logger.debug(f"Running custom analysis script: {script_path}")

            # Import the custom script as a module
            spec = importlib.util.spec_from_file_location(
                "custom_analysis", script_path
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Look for a run_analysis function
            if hasattr(custom_module, "run_analysis"):
                # Call with pipeline manager context
                custom_module.run_analysis(
                    model=getattr(self, "model", None),
                    config=self.config_instance,
                    path_manager=getattr(self, "path_manager", None),
                    pipeline=self,
                )
                logger.info("Custom analysis script completed successfully.")
            else:
                logger.error(
                    "Custom analysis script must define a 'run_analysis' function"
                )

        except Exception as e:
            logger.error(f"Error running custom analysis script: {e}")
            # Don't raise - analysis scripts are optional

    def run_evaluation(self, skip_setup=False) -> dict[str, Any]:
        """
        Run evaluation-only pipeline that uses pre-trained model on holdout data.

        Args:
            skip_setup: Skip pipeline setup if already done (for combined pipeline)

        Returns:
            Dict containing model path and results path
        """
        import time

        if not skip_setup:
            self._setup_pipeline()

        # Load trained model (components + model) and preprocess only eval data
        self.load_model()
        self.preprocess_data(for_evaluation=True)

        # Start evaluation timing after data loading/preprocessing
        evaluation_start_time = time.time()

        # Run evaluation metrics (standalone evaluation: save to recent directory only)
        self.run_metrics("recent")

        # Analysis (conditional based on config)
        if self.config_instance.interpretation.shap.enabled:
            self.run_interpretation()

        if self.config_instance.visualizations.enabled:
            self.run_visualizations()

        # Run project-specific analysis script if enabled
        if self.config_instance.analysis.run_analysis_script.enabled:
            self.run_analysis_script()

        # Calculate evaluation duration
        evaluation_duration = time.time() - evaluation_start_time

        # Return paths for CLI feedback
        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
        return {
            "model_path": experiment_dir,
            "results_path": os.path.join(experiment_dir, "evaluation"),
            "evaluation_duration": evaluation_duration,
        }

    def run_pipeline(self) -> dict[str, Any]:
        """
        Run the complete pipeline: training + evaluation.
        Handles setup once, then runs both phases efficiently.

        Returns:
            Dict containing model path and results path
        """
        # Mark this as a combined pipeline to preserve data for evaluation
        self._in_combined_pipeline = True

        # Setup phases (once for entire pipeline)
        self._setup_pipeline()

        # Step 1: Training (skip setup since we already did it)
        training_results = self.run_training(skip_setup=True)

        # Step 2: Evaluation (skip setup since we already did it)
        try:
            evaluation_results = self.run_evaluation(skip_setup=True)
        except Exception as e:
            logger.warning(f"Automatic post-training evaluation failed: {e}")
            logger.info("You can run 'timeflies evaluate' manually later")
            evaluation_results = {}

        # Combine results
        pipeline_results = {
            **training_results,
            **evaluation_results,
            "model_improved": self.model_improved,
        }

        # Add best symlink info if model improved
        if self.model_improved:
            pipeline_results["best_results_path"] = os.path.join(
                training_results.get("model_path", ""), "evaluation"
            )

        # Update best/latest folders again to include evaluation results
        if evaluation_results:  # Only update if evaluation succeeded
            if self.model_improved:
                self.path_manager.update_best_folder(self.experiment_name)
            self.path_manager.update_latest_folder(self.experiment_name)

        # Update timing information in final summary if evaluation completed
        if evaluation_results and "evaluation_duration" in evaluation_results:
            self._print_final_timing_summary(
                evaluation_results.get("evaluation_duration", 0)
            )

        # Final cleanup after combined pipeline
        self._cleanup_memory(keep_eval_data=False, keep_vis_data=False)
        self._in_combined_pipeline = False

        return pipeline_results
