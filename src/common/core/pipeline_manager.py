import gc
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

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
        print("\n" + "=" * 60)
        print("Data Overview")
        print("=" * 60)

        # Configuration details
        target = getattr(self.config_instance.data, "target_variable", "age")
        tissue = getattr(self.config_instance.data, "tissue", "unknown")
        batch_corrected = getattr(
            self.config_instance.data.batch_correction, "enabled", False
        )

        print(f"Target Variable:        {target.title()}")
        print(f"Tissue Type:            {tissue.title()}")
        print(f"Batch Corrected:        {'Yes' if batch_corrected else 'No'}")

        # Split configuration
        from common.utils.split_naming import SplitNamingUtils

        split_config = SplitNamingUtils.extract_split_details_for_metadata(
            self.config_instance
        )
        if split_config:
            print(
                f"Split Method:           {split_config.get('method', 'unknown').title()}"
            )
            if split_config.get("method") == "column":
                print(
                    f"Split Column:           {split_config.get('column', 'unknown')}"
                )
                train_vals = split_config.get("train_values", [])
                test_vals = split_config.get("test_values", [])
                if train_vals:
                    print(f"Training Values:        {', '.join(train_vals)}")
                if test_vals:
                    print(f"Evaluation Values:      {', '.join(test_vals)}")

        print("=" * 60)

    def _print_training_and_evaluation_data(self):
        """Print training and evaluation data details after preprocessing."""
        if not (hasattr(self, "train_data") and hasattr(self, "test_data")):
            return

        encoding_var = getattr(self.config_instance.data, "target_variable", "age")

        print("\n📊 TRAINING DATA:")
        print("-" * 60)

        # Get training data info
        train_cells = (
            self.train_data.shape[0] if hasattr(self.train_data, "shape") else 0
        )
        train_genes = (
            self.train_data.shape[1] if hasattr(self.train_data, "shape") else 0
        )
        print(
            f"Training Data (Preprocessed): {train_cells:,} cells, {train_genes:,} genes"
        )

        # Add space then show age distribution
        if hasattr(self, "train_labels") and hasattr(self, "label_encoder"):
            print(f"\n{encoding_var.title()} Distribution (Training Data):")
            import numpy as np

            unique, counts = np.unique(self.train_labels, return_counts=True)
            total = counts.sum()
            for label_encoded, count in zip(unique, counts):
                # Decode the label if possible
                try:
                    original_label = self.label_encoder.inverse_transform(
                        [label_encoded]
                    )[0]
                    pct = (count / total) * 100
                    unit = "days" if encoding_var == "age" else ""
                    print(
                        f"  └─ {original_label} {unit:5s}: {count:6,} samples ({pct:5.1f}%)"
                    )
                except Exception:
                    pass

        # Show evaluation data if available
        if hasattr(self, "test_data") and self.test_data.shape[0] > 0:
            eval_cells = self.test_data.shape[0]
            eval_genes = self.test_data.shape[1]
            print("\nDataset Size:")
            print(f"  └─ Samples:           {eval_cells:,}")
            print(f"  └─ Features (genes):  {eval_genes:,}")

            print("\nClass Distribution:")
            if hasattr(self, "test_labels") and hasattr(self, "label_encoder"):
                unique, counts = np.unique(self.test_labels, return_counts=True)
                total = counts.sum()
                for label_encoded, count in zip(unique, counts):
                    try:
                        original_label = self.label_encoder.inverse_transform(
                            [label_encoded]
                        )[0]
                        pct = (count / total) * 100
                        print(
                            f"  └─ {original_label:10s} : {count:6,} samples ({pct:5.1f}%)"
                        )
                    except Exception:
                        pass

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
            ) = self.data_preprocessor.prepare_data()
            # Clean up memory (preserve evaluation data for auto-evaluation)
            # During combined pipeline, also preserve visualization data for post-training evaluation
            is_combined_pipeline = getattr(self, "_in_combined_pipeline", False)
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

    def train_model(self):
        """
        Preprocess data and build/train a new model.
        """
        # Build and train the model
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

        # History and model improvement tracked internally
        self.model, self.history, self.model_improved = self.model_builder.run()

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
                location = (
                    f"{result_type} directory"
                    if result_type
                    else "experiment directory"
                )
                logger.info(
                    f"Training visualizations saved successfully to {location}."
                )

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
        if not skip_setup:
            self._print_header("🔥 TRAINING PIPELINE")
            self._setup_pipeline()

        # Preprocessing
        self.preprocess_data()

        # Display training and evaluation data
        self._print_training_and_evaluation_data()

        # Model training
        print("\nTraining Progress:")
        print("-" * 60)
        import time

        self._training_start_time = time.time()
        self.train_model()

        # Save outputs and create symlinks after training
        self.save_outputs(metadata=True, symlinks=True)

        # Print training completion summary with improvement status
        improvement_status = "🆕 NEW BEST" if self.model_improved else "📊 BASELINE"
        print(f"\n✅ TRAINING COMPLETED - {improvement_status}")

        # Training Summary section with double lines
        print("\n" + "=" * 60)
        print("📋 TRAINING SUMMARY")
        print("=" * 60)

        # Configuration info
        from common.utils.split_naming import SplitNamingUtils

        split_config = SplitNamingUtils.extract_split_details_for_metadata(
            self.config_instance
        )
        if split_config and split_config.get("method") == "column":
            train_vals = split_config.get("train_values", [])
            test_vals = split_config.get("test_values", [])
            split_info = f"Split: {'-vs-'.join(train_vals + test_vals)} (column)"
            print(split_info)

        model_type = getattr(self.config_instance.data, "model", "CNN")
        target = getattr(self.config_instance.data, "target_variable", "age")
        tissue = getattr(self.config_instance.data, "tissue", "head")
        print(f"Model: {model_type}, Target: {target}, Tissue: {tissue}")

        # Training results
        if hasattr(self, "history") and self.history:
            val_losses = self.history.history.get("val_loss", [])
            if val_losses:
                best_epoch = val_losses.index(min(val_losses)) + 1
                best_val_loss = min(val_losses)
                print(f"Best Epoch: {best_epoch}")
                print(f"Best Val Loss: {best_val_loss:.4f}")

        print(f"Model saved to: {self.experiment_name}")
        print("=" * 60)

        # Show training duration if available
        import time

        if hasattr(self, "_training_start_time"):
            duration = time.time() - self._training_start_time
            print(f"Training duration: {duration:.1f} seconds")
        print("-" * 60)

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
        if not skip_setup:
            self._setup_pipeline()

        # Load trained model (components + model) and preprocess only eval data
        self.load_model()
        self.preprocess_data(for_evaluation=True)

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

        # Return paths for CLI feedback
        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
        return {
            "model_path": experiment_dir,
            "results_path": os.path.join(experiment_dir, "evaluation"),
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

        # Show project and dataset overview after setup
        self._print_project_and_dataset_overview()

        # Step 1: Training (skip setup since we already did it)
        training_results = self.run_training(skip_setup=True)

        # Step 2: Evaluation (skip setup since we already did it)
        self._print_header(f"🧪 HOLDOUT EVALUATION - {self.experiment_name}")
        print("-" * 60)

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

        # Final cleanup after combined pipeline
        self._cleanup_memory(keep_eval_data=False, keep_vis_data=False)
        self._in_combined_pipeline = False

        # Final summary section removed as requested

        return pipeline_results
