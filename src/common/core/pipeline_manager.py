import importlib.util
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# Suppress TensorFlow and gRPC logging
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common.data.loaders import DataLoader
from common.data.preprocessing.data_processor import DataPreprocessor
from common.data.preprocessing.gene_filter import GeneFilter
from common.display.display_manager import DisplayManager

# ModelBuilder, ModelLoader now handled by ModelManager
from common.utils.gpu_handler import GPUHandler
from common.utils.path_manager import PathManager
from common.utils.storage_manager import StorageManager

from .config_manager import Config
from .model_manager import ModelManager

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
        self.display_manager = DisplayManager(self.config_instance)
        self.model_manager = ModelManager(self.config_instance, self.path_manager, mode)
        self.mode = mode

        # Set experiment name based on mode
        if mode == "training":
            # Generate new experiment for training
            self.experiment_name = self.path_manager.generate_experiment_name()
        else:
            # Evaluation mode - find model using training key (compatible model)
            try:
                # Check if a compatible trained model exists using training key
                models_dir = self.path_manager.get_models_folder_path()

                if Path(models_dir).exists():
                    # Create new experiment name for this evaluation run
                    self.experiment_name = self.path_manager.generate_experiment_name()
                    logger.info(f"Using trained model from: {models_dir}")

                    # Store models_dir for later copying after evaluation
                    self._models_dir_for_copying = models_dir
                else:
                    training_key = self.path_manager.get_config_key().split("-vs-")[0]
                    raise RuntimeError(
                        f"No trained model found with training key: {training_key}"
                    )

            except (FileNotFoundError, RuntimeError) as e:
                raise RuntimeError(
                    f"No trained model found for evaluation. Please train a model first. Error: {e}"
                )

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

        # Initialize combined pipeline flag
        self._in_combined_pipeline = False

        # Initializing TimeFlies pipeline

    def _setup_pipeline(self):
        """Common pipeline setup: GPU, data loading, gene filtering, and evaluation data preparation."""
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

        # Always prepare evaluation data regardless of mode
        # This ensures consistent data for both training+eval and eval-only modes
        if self.adata_eval is not None:
            # Create data preprocessor for evaluation data preparation
            eval_preprocessor = DataPreprocessor(
                self.config_instance, self.adata, self.adata_corrected
            )

            # Use corrected data if available, otherwise regular data
            adata_eval_to_use = self.adata_eval_corrected or self.adata_eval

            # For training mode, we'll fit components first, then prepare eval data
            # For eval-only mode, we'll use loaded components
            if self.mode == "training":
                # In training mode, we'll prepare eval data after fitting training components
                self._eval_preprocessor = eval_preprocessor
                self._adata_eval_to_use = adata_eval_to_use
            else:
                # In evaluation-only mode, prepare eval data now with loaded components
                # Note: components will be loaded separately in run_evaluation
                self._eval_preprocessor = eval_preprocessor
                self._adata_eval_to_use = adata_eval_to_use

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
            is_combined_pipeline = getattr(self, "_in_combined_pipeline", False)
            if (
                is_combined_pipeline
                and hasattr(self, "eval_data")
                and self.eval_data is not None
            ):
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
            # Memory cleanup removed - causes problems
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

            # Always process evaluation data with fitted components from training
            if (
                hasattr(self, "_adata_eval_to_use")
                and self._adata_eval_to_use is not None
            ):
                (
                    self.eval_data,
                    self.eval_labels,
                    _,  # We already have label encoder
                    self.eval_adata,  # Filtered AnnData for display/metrics
                ) = self.data_preprocessor.prepare_final_eval_data(
                    self._adata_eval_to_use,
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
                self.eval_adata = None

            # Memory cleanup removed - causes problems

    def run_training(self) -> dict[str, Any]:
        """
        Run the complete training pipeline: setup + preprocessing + model training.

        Returns:
            Dict containing training results and paths
        """

        self._setup_pipeline()

        # Start training timer to include preprocessing
        self._training_start_time = time.time()

        # Preprocessing with timing
        preprocessing_start_time = time.time()
        self.preprocess_data()
        self._preprocessing_duration = time.time() - preprocessing_start_time

        # Display training and evaluation data
        # Show training and evaluation data using DisplayManager
        try:
            self.display_manager.print_training_and_evaluation_data(
                getattr(self, "train_data", None),
                getattr(self, "eval_data", None),
                self.config_instance,
                train_subset=getattr(self, "train_subset", None),
                eval_subset=getattr(
                    self, "eval_adata", None
                ),  # Use the filtered and processed adata for display
            )
        except Exception as e:
            logger.warning(f"Could not display training and evaluation data: {e}")

        # Build and train model using ModelManager
        self.model, self.model_builder = self.model_manager.build_model(
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

        # Display model architecture after building but before training
        self.display_manager.display_model_architecture(
            self.model, self.config_instance
        )

        # Model training
        previous_best_msg = self.model_manager.get_previous_best_loss_message(
            self.experiment_name
        )
        self.display_manager.print_training_progress_header(previous_best_msg)

        # Store original previous best loss for comparison (before training updates it)
        self._original_previous_best_loss = (
            self.model_manager.get_previous_best_validation_loss(self.experiment_name)
        )

        self.history, self.model, self.model_improved = self.model_manager.train_model(
            self.model_builder, self.model
        )

        # Set num_features and gene names for auto-evaluation
        # Get number of genes (features) - handle 3D CNN data shape (cells, 1, genes)
        if len(self.train_data.shape) == 3:
            self.num_features = self.train_data.shape[2]  # genes in last dimension
        else:
            self.num_features = self.train_data.shape[1]  # standard 2D data
        # Store the exact gene names used in training for consistent evaluation
        if hasattr(self.data_preprocessor, "train_gene_names"):
            self.train_gene_names = self.data_preprocessor.train_gene_names
        else:
            self.train_gene_names = None

        # Save training visuals to experiment directory
        if self.config_instance.visualizations.enabled:
            # Generating training visualizations
            self.storage_manager.save_outputs(
                self, training_visuals=True
            )  # Save to experiment directory

        # Save outputs and create symlinks after training
        self.storage_manager.save_outputs(self, metadata=True, symlinks=True)

        # Results section
        improvement_status = (
            "New best model found"
            if self.model_improved
            else "No improvement over existing model found"
        )

        # Store training duration for later use
        if hasattr(self, "_training_start_time"):
            training_duration = time.time() - self._training_start_time
            self._stored_training_duration = training_duration

        # Use display manager for training results
        original_previous_best_loss = getattr(
            self, "_original_previous_best_loss", None
        )
        history = getattr(self, "history", None)
        self.display_manager.print_training_results(
            history,
            original_previous_best_loss,
            self.experiment_name,
            improvement_status,
        )

        # Return training results
        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
        return {
            "model_path": experiment_dir,
            "experiment_name": self.experiment_name,
        }

        # Don't raise - analysis scripts are optional

    def run_evaluation(self, skip_setup=False, is_training=False) -> dict[str, Any]:
        """
        Run evaluation-only pipeline that uses pre-trained model on holdout data.

        Args:
            skip_setup: Skip pipeline setup if already done (for combined pipeline)
            is_training: Flag indicating if called during training (uses in-memory model) vs standalone (loads from disk)

        Returns:
            Dict containing model path and results path
        """

        if not skip_setup:
            # Time the preprocessing/setup phase for standalone evaluation
            preprocessing_start_time = time.time()
            self._setup_pipeline()

            # Only load model in standalone evaluation mode
            # In combined pipeline during training, model is already in memory
            if not is_training:
                # Load model using ModelManager
                self.model, components, self.model_loader = (
                    self.model_manager.load_model_components(self.config_instance)
                )
                # Unpack components
                (
                    self.label_encoder,
                    self.scaler,
                    self.is_scaler_fit,
                    self.highly_variable_genes,
                    self.num_features,
                    self.history,
                    self.mix_included,
                    self.reference_data,
                ) = components

                # Now prepare evaluation data with loaded components
                if (
                    hasattr(self, "_adata_eval_to_use")
                    and self._adata_eval_to_use is not None
                ):
                    (
                        self.eval_data,
                        self.eval_labels,
                        _,  # We already have label encoder
                        self.eval_adata,  # Filtered AnnData for display/metrics
                    ) = self._eval_preprocessor.prepare_final_eval_data(
                        self._adata_eval_to_use,
                        self.label_encoder,
                        self.num_features,
                        self.scaler,
                        self.is_scaler_fit,
                        self.highly_variable_genes,
                        self.mix_included,
                        getattr(self._eval_preprocessor, "train_gene_names", None),
                    )

                # Set test_data and test_labels for compatibility with existing methods
                self.test_data = self.eval_data
                self.test_labels = self.eval_labels

            # Capture preprocessing duration for standalone evaluation
            self._preprocessing_duration = time.time() - preprocessing_start_time

        # Ensure evaluation data is properly assigned for combined pipeline
        if skip_setup and is_training:
            # In combined pipeline, use data prepared during training
            self.test_data = getattr(self, "eval_data", None)
            self.test_labels = getattr(self, "eval_labels", None)
        elif not hasattr(self, "test_data") or self.test_data is None:
            # Fallback assignment
            self.test_data = self.eval_data
            self.test_labels = self.eval_labels

        # Only show evaluation data overview and model architecture in standalone evaluation
        # In combined pipeline, this info is already shown during training
        if not skip_setup:  # standalone evaluation mode
            # Display preprocessed data overview using pre-prepared data
            try:
                self.display_manager.print_evaluation_data(
                    self.eval_data,
                    self.config_instance,
                    eval_subset=getattr(self, "eval_adata", None),
                )
            except Exception as e:
                logger.warning(f"Could not display evaluation data: {e}")

            # Display model architecture (similar to training pipeline)
            self.display_manager.display_model_architecture(
                getattr(self, "model", None), self.config_instance
            )

        # Start evaluation timing after data loading/preprocessing
        evaluation_start_time = time.time()

        # Run evaluation metrics using the actual loaded experiment name
        # This ensures results are saved to the correct experiment directory
        self.run_metrics("best" if not skip_setup else "recent")

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

        # Update folders after evaluation
        if not skip_setup:  # Only in standalone evaluation mode
            # Always update latest folder to reflect recent activity
            self.path_manager.update_latest_folder(self.experiment_name)

            # If this is the best model, also update best folder with new evaluation results
            try:
                best_experiment_name = self.path_manager.get_best_experiment_name()
                if (
                    best_experiment_name
                    and best_experiment_name == self.experiment_name
                ):
                    self.path_manager.update_best_folder(self.experiment_name)
            except Exception:
                # If we can't determine best experiment, skip updating best folder
                pass

        # Return paths for CLI feedback
        experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)

        # Display timing appropriate for the mode
        if not skip_setup:  # Standalone evaluation mode
            preprocessing_duration = getattr(self, "_preprocessing_duration", 0)
            self.display_manager.print_timing_summary(
                preprocessing_duration=preprocessing_duration,
                evaluation_duration=evaluation_duration,
            )
        # In combined pipeline, timing is handled by run_pipeline()

        # Create eval metadata in evaluation folder
        self._create_eval_metadata(experiment_dir, evaluation_duration)

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

        # Step 1: Training (includes its own setup)
        training_results = self.run_training()

        # Step 2: Evaluation (skip setup since we already did it, use in-memory model)
        try:
            evaluation_results = self.run_evaluation(skip_setup=True, is_training=True)
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

        # Note: best/latest folder updates are handled by storage_manager.save_outputs()
        # to avoid duplicate calls and ensure proper coordination

        # Get stored durations from different phases
        preprocessing_duration = getattr(self, "_preprocessing_duration", 0)
        training_duration = getattr(self, "_stored_training_duration", 0)
        evaluation_duration = evaluation_results.get("evaluation_duration", 0)

        # Use display manager's timing summary
        self.display_manager.print_timing_summary(
            preprocessing_duration, training_duration, evaluation_duration
        )

        # Memory cleanup removed - causes problems
        self._in_combined_pipeline = False

        return pipeline_results

    def run_metrics(self, result_type="recent"):
        """
        Run evaluation metrics using the Metrics class.

        Args:
            result_type: "recent" or "best" for output directory selection
        """
        if self.metrics_class is None:
            logger.warning("Metrics class not available, skipping metrics evaluation")
            return

        # Determine pipeline mode - don't show eval info if this is combined pipeline
        is_combined_pipeline = getattr(self, "_in_combined_pipeline", False)
        pipeline_mode = "training" if is_combined_pipeline else "evaluation"

        # Get the correct output directory for this experiment
        evaluation_output_dir = self.path_manager.get_experiment_evaluation_dir(
            self.experiment_name
        )

        # Create metrics evaluator
        metrics_evaluator = self.metrics_class(
            config=self.config_instance,
            model=self.model,
            test_data=self.test_data,
            test_labels=self.test_labels,
            label_encoder=self.label_encoder,
            path_manager=self.path_manager,
            result_type=result_type,
            output_dir=evaluation_output_dir,  # Pass the specific experiment evaluation directory
            pipeline_mode=pipeline_mode,
        )

        # Run metrics evaluation
        if hasattr(metrics_evaluator, "compute_metrics"):
            metrics_evaluator.compute_metrics()
        elif hasattr(metrics_evaluator, "run_evaluation"):
            metrics_evaluator.run_evaluation()
        elif hasattr(metrics_evaluator, "run"):
            metrics_evaluator.run()
        else:
            logger.warning("Metrics evaluator does not have a run method")

    def run_interpretation(self):
        """Run SHAP interpretation if enabled and available."""
        if self.interpreter_class is None:
            logger.warning("Interpreter class not available, skipping interpretation")
            return

        # Get the correct output directory for this experiment
        evaluation_output_dir = self.path_manager.get_experiment_evaluation_dir(
            self.experiment_name
        )

        # Create interpreter
        interpreter = self.interpreter_class(
            config=self.config_instance,
            model=self.model,
            test_data=self.test_data,
            test_labels=self.test_labels,
            label_encoder=self.label_encoder,
            reference_data=self.test_data,
            path_manager=self.path_manager,
        )

        # Run interpretation
        if hasattr(interpreter, "run"):
            interpreter.run()
        else:
            logger.warning("Interpreter does not have a run method")

    def run_visualizations(self):
        """Run visualizations if enabled and available."""
        if self.visualizer_class is None:
            logger.warning("Visualizer class not available, skipping visualizations")
            return

        # Use storage manager to handle visualizations
        self.storage_manager.save_outputs(self, evaluation_visuals=True)

        # For eval-only runs, copy training artifacts AFTER evaluation is complete
        if hasattr(self, "_models_dir_for_copying"):
            self._copy_training_artifacts_for_evaluation(self._models_dir_for_copying)

        # Copy evaluation results to best/ and latest/ if this experiment improved the model
        if (
            hasattr(self, "model_improved")
            and self.model_improved
            and self.experiment_name
        ):
            self._copy_evaluation_to_symlink_folders()

        # For eval-only runs, check if this is the first experiment in this config
        # If so, it should become the "best" for this config
        is_combined_pipeline = getattr(self, "_in_combined_pipeline", False)
        if not is_combined_pipeline:  # This is eval-only mode
            self._handle_eval_only_best_folder()
            # Ensure latest/ always has model symlink for eval-only runs
            self._ensure_latest_model_symlink()

    def _copy_training_artifacts_for_evaluation(self, models_dir: str):
        """Copy training artifacts from original best experiment for eval-only runs."""
        try:
            import os
            import shutil
            from pathlib import Path

            # Get target experiment directory
            experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Find the original training experiment (the one that created this model)
            training_key = self.path_manager.get_config_key().split("-vs-")[0]

            # Look for existing best/ directory with the same training key
            best_experiment = None
            models_parent = Path(models_dir).parent  # .../models/
            experiments_parent = models_parent.parent  # .../classification/
            best_dir = experiments_parent / "best"

            if best_dir.exists():
                # Find the best experiment directory with matching training key
                for item in best_dir.iterdir():
                    if item.is_dir() and item.name.startswith(training_key):
                        best_experiment = item
                        break

            if best_experiment and best_experiment.exists():
                # Copy ALL artifacts from the best experiment except evaluation/
                for item in best_experiment.iterdir():
                    if item.name == "evaluation":
                        continue  # Skip evaluation, we'll create our own

                    target_path = os.path.join(experiment_dir, item.name)
                    if item.is_dir():
                        shutil.copytree(item, target_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target_path)

                    logger.debug(
                        f"Copied training artifacts from {best_experiment} to {experiment_dir}"
                    )

                # Update metadata to reflect current evaluation configuration
                self._update_eval_metadata(experiment_dir, str(best_experiment))

            else:
                # Fallback: just copy model files
                model_components_dir = os.path.join(experiment_dir, "model_components")
                os.makedirs(model_components_dir, exist_ok=True)

                models_path = Path(models_dir)
                for item in models_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, model_components_dir)

                logger.warning(
                    "Could not find original training experiment, copied model files only"
                )

        except Exception as e:
            logger.warning(f"Could not copy training artifacts: {e}")

    def _update_eval_metadata(self, experiment_dir: str, source_experiment: str):
        """Update metadata.json to reflect current evaluation configuration."""
        try:
            import json
            from datetime import datetime

            metadata_path = os.path.join(experiment_dir, "metadata.json")

            if os.path.exists(metadata_path):
                # Load existing metadata
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Update with current evaluation info
                metadata.update(
                    {
                        "experiment_name": self.experiment_name,
                        "created_timestamp": datetime.now().isoformat(),
                        "evaluation_completed_at": datetime.now().isoformat(),
                        "evaluation_count": 1,
                        "source_experiment": source_experiment,
                        "experiment_type": "evaluation_only",
                        "split_config": {
                            "method": "column",
                            "split_name": self.path_manager.get_config_key().split("_")[
                                -1
                            ],  # Get split part
                            "column": getattr(
                                self.config_instance.data.split, "column", "unknown"
                            ),
                            "train_values": getattr(
                                self.config_instance.data.split, "train", []
                            ),
                            "test_values": getattr(
                                self.config_instance.data.split, "test", []
                            ),
                        },
                    }
                )

                # Write updated metadata
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.debug(
                    f"Updated metadata for eval-only experiment: {self.experiment_name}"
                )

        except Exception as e:
            logger.warning(f"Could not update eval metadata: {e}")

    def _create_eval_metadata(self, experiment_dir: str, evaluation_duration: float):
        """Create eval_metadata.json in the evaluation folder with evaluation details."""
        try:
            import json
            from datetime import datetime
            
            eval_dir = os.path.join(experiment_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            
            eval_metadata_path = os.path.join(eval_dir, "eval_metadata.json")
            
            # Get data shapes if pipeline has data
            data_shapes = {}
            if hasattr(self, 'test_data') and self.test_data is not None:
                data_shapes["test_samples"] = self.test_data.shape[0]
                
                # Always use num_features (original feature count) if available
                if hasattr(self, 'num_features') and self.num_features:
                    data_shapes["test_features"] = self.num_features
                else:
                    # Try to get original features from training experiment metadata
                    try:
                        best_exp_dir = self.path_manager.get_best_experiment_dir()
                        metadata_path = os.path.join(best_exp_dir, "metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                training_metadata = json.load(f)
                                if "data_shapes" in training_metadata and "n_features" in training_metadata["data_shapes"]:
                                    data_shapes["test_features"] = training_metadata["data_shapes"]["n_features"]
                                else:
                                    # Final fallback to test data shape
                                    data_shapes["test_features"] = self.test_data.shape[1]
                        else:
                            # Fallback: get gene count from test data shape
                            if len(self.test_data.shape) == 3:
                                data_shapes["test_features"] = self.test_data.shape[2]  # (cells, 1, genes)
                            else:
                                data_shapes["test_features"] = self.test_data.shape[1]
                    except Exception:
                        # Final fallback to test data shape - check for 3D CNN data
                        if len(self.test_data.shape) == 3:
                            data_shapes["test_features"] = self.test_data.shape[2]  # (cells, 1, genes)
                        else:
                            data_shapes["test_features"] = self.test_data.shape[1]
                    
            # Get split configuration
            split_config = {
                "method": getattr(self.config_instance.data.split, "method", "unknown"),
                "column": getattr(self.config_instance.data.split, "column", "unknown"),
                "train_values": getattr(self.config_instance.data.split, "train", []),
                "test_values": getattr(self.config_instance.data.split, "test", [])
            }
            
            # Create eval metadata
            eval_metadata = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_duration_seconds": evaluation_duration,
                "experiment_name": self.experiment_name,
                "model_type": self.config_instance.data.model,
                "target_variable": self.config_instance.data.target_variable,
                "tissue": self.config_instance.data.tissue,
                "data_filters": {
                    "sex": getattr(self.config_instance.data, "sex", "unknown"),
                    "cell_type": getattr(self.config_instance.data.cell, "type", "unknown") if hasattr(self.config_instance.data, "cell") else "unknown",
                    "cell_column": getattr(self.config_instance.data.cell, "column", "unknown") if hasattr(self.config_instance.data, "cell") else "unknown"
                },
                "split_config": split_config,
                "data_shapes": data_shapes,
                "batch_correction": getattr(self.config_instance.data.batch_correction, "enabled", False),
                "evaluation_settings": {
                    "metrics_enabled": True,
                    "visualizations_enabled": self.config_instance.visualizations.enabled,
                    "shap_enabled": self.config_instance.interpretation.shap.enabled,
                    "analysis_script_enabled": self.config_instance.analysis.run_analysis_script.enabled
                }
            }
            
            # Save eval metadata
            with open(eval_metadata_path, "w") as f:
                json.dump(eval_metadata, f, indent=2)
                
            logger.debug(f"Created eval metadata: {eval_metadata_path}")
            
        except Exception as e:
            logger.warning(f"Could not create eval metadata: {e}")

    def _same_evaluation_conditions(self, current_config, best_metadata: dict) -> bool:
        """Check if current run has same evaluation conditions as existing best."""
        try:
            # Compare test values and column
            current_test = set(getattr(current_config.data.split, "test", []))
            current_column = getattr(current_config.data.split, "column", "")

            best_split = best_metadata.get("split_config", {})
            best_test = set(best_split.get("test_values", []))
            best_column = best_split.get("column", "")

            return current_test == best_test and current_column == best_column

        except Exception as e:
            logger.warning(f"Could not compare evaluation conditions: {e}")
            return False

    def _get_evaluation_metrics(self, experiment_dir: str) -> dict:
        """Extract evaluation metrics from experiment directory."""
        try:
            import json

            # Try to get metrics from evaluation results
            eval_dir = os.path.join(experiment_dir, "evaluation")
            if os.path.exists(eval_dir):
                # Look for metrics files (could be various formats)
                for filename in ["metrics.json", "evaluation_results.json"]:
                    metrics_path = os.path.join(eval_dir, filename)
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            return json.load(f)

            return {}
        except Exception as e:
            logger.warning(f"Could not extract evaluation metrics: {e}")
            return {}

    def _should_update_best_based_on_metrics(
        self, current_experiment_dir: str, best_metadata: dict
    ) -> bool:
        """Compare current run metrics with existing best to determine if should update."""
        try:
            # Get current run metrics
            current_metrics = self._get_evaluation_metrics(current_experiment_dir)

            # Get best run metrics - could be from metadata or evaluation files
            best_metrics = best_metadata.get("evaluation", {})
            if not best_metrics and "best_val_loss" in best_metadata.get(
                "training", {}
            ):
                # Use training metrics as fallback
                best_metrics = {"val_loss": best_metadata["training"]["best_val_loss"]}

            if not current_metrics or not best_metrics:
                logger.warning(
                    "Could not extract metrics for comparison, defaulting to update"
                )
                return True

            # Compare primary metric (accuracy for classification, loss for others)
            primary_metric = self._get_primary_metric(current_metrics, best_metrics)

            if primary_metric:
                current_value = current_metrics.get(primary_metric)
                best_value = best_metrics.get(primary_metric)

                if current_value is not None and best_value is not None:
                    # For accuracy/f1: higher is better, for loss: lower is better
                    if primary_metric in ["accuracy", "f1_score", "auc"]:
                        return current_value > best_value
                    elif primary_metric in ["loss", "val_loss"]:
                        return current_value < best_value

            # Default: update if we can't determine (keeps system moving forward)
            return True

        except Exception as e:
            logger.warning(f"Could not compare metrics: {e}")
            return True

    def _get_primary_metric(self, current_metrics: dict, best_metrics: dict) -> str:
        """Determine primary metric to use for comparison."""
        # Priority order for metrics
        metric_priority = ["accuracy", "f1_score", "auc", "val_loss", "loss"]

        for metric in metric_priority:
            if metric in current_metrics and metric in best_metrics:
                return metric

        return None

    def _update_best_rerun_timestamp(self, best_dir: str):
        """Update existing best metadata with rerun timestamp."""
        try:
            import json
            from datetime import datetime

            metadata_path = os.path.join(best_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Add rerun info
                if "reruns" not in metadata:
                    metadata["reruns"] = []

                metadata["reruns"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "experiment_name": self.experiment_name,
                        "note": "Same config rerun, performance not improved",
                    }
                )

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.debug("Updated best metadata with rerun timestamp")

        except Exception as e:
            logger.warning(f"Could not update rerun timestamp: {e}")

    def _handle_eval_only_best_folder(self):
        """Handle best/ folder updates for eval-only runs based on evaluation conditions."""
        try:
            import json

            best_dir = self.path_manager.get_best_folder_path()

            if not os.path.exists(best_dir):
                # No best exists for this config, make this the first best
                logger.info("Creating best/ directory for new config (eval-only run)")
                self.path_manager.update_best_folder(self.experiment_name)
                # Update metadata in best/ folder with current evaluation configuration
                best_experiment_dir = self.path_manager.get_best_folder_path()
                current_experiment_dir = self.path_manager.get_experiment_dir(
                    self.experiment_name
                )
                self._update_eval_metadata(
                    best_experiment_dir, str(current_experiment_dir)
                )

            else:
                # Best exists, check if we should compete or update
                best_metadata_path = os.path.join(best_dir, "metadata.json")

                if os.path.exists(best_metadata_path):
                    with open(best_metadata_path) as f:
                        best_metadata = json.load(f)

                    # Check if evaluation conditions are the same
                    if self._same_evaluation_conditions(
                        self.config_instance, best_metadata
                    ):
                        # Same conditions - compete for best based on metrics!
                        current_experiment_dir = self.path_manager.get_experiment_dir(
                            self.experiment_name
                        )

                        if self._should_update_best_based_on_metrics(
                            current_experiment_dir, best_metadata
                        ):
                            logger.info(
                                "New best performance! Updating best/ directory"
                            )
                            self.path_manager.update_best_folder(self.experiment_name)
                            # Update metadata in best/ folder with current evaluation configuration
                            best_experiment_dir = (
                                self.path_manager.get_best_folder_path()
                            )
                            self._update_eval_metadata(
                                best_experiment_dir, str(current_experiment_dir)
                            )
                        else:
                            logger.info(
                                "Current performance not better than existing best, keeping existing best/"
                            )
                            # Still update metadata with rerun timestamp in existing best
                            self._update_best_rerun_timestamp(best_dir)

                    else:
                        # Different conditions - keep existing best, don't update
                        logger.debug(
                            "Different evaluation conditions, keeping existing best/"
                        )
                else:
                    # Best exists but no metadata, update to be safe
                    logger.info("Updating best/ directory (no metadata found)")
                    self.path_manager.update_best_folder(self.experiment_name)
                    # Update metadata in best/ folder with current evaluation configuration
                    best_experiment_dir = self.path_manager.get_best_folder_path()
                    current_experiment_dir = self.path_manager.get_experiment_dir(
                        self.experiment_name
                    )
                    self._update_eval_metadata(
                        best_experiment_dir, str(current_experiment_dir)
                    )

        except Exception as e:
            logger.warning(f"Could not handle eval-only best folder: {e}")

    def _ensure_latest_model_symlink(self):
        """Ensure latest/ has model_components symlink for eval-only runs."""
        try:
            latest_dir = self.path_manager.get_latest_folder_path()
            models_dir = self.path_manager.get_models_folder_path()

            if os.path.exists(latest_dir) and os.path.exists(models_dir):
                latest_symlink = os.path.join(latest_dir, "model_components")

                # Remove existing model_components if present
                if os.path.exists(latest_symlink):
                    if os.path.islink(latest_symlink):
                        os.unlink(latest_symlink)
                    elif os.path.isdir(latest_symlink):
                        import shutil

                        shutil.rmtree(latest_symlink)

                # Create symlink to models/
                try:
                    relative_models_path = os.path.relpath(models_dir, latest_dir)
                    os.symlink(relative_models_path, latest_symlink)
                    logger.debug("Ensured model symlink in latest/ for eval-only run")
                except (OSError, NotImplementedError) as e:
                    logger.warning(f"Could not create latest symlink: {e}")

        except Exception as e:
            logger.warning(f"Could not ensure latest model symlink: {e}")

    def _copy_evaluation_to_symlink_folders(self):
        """Copy evaluation results (plots, metrics) to best/ and latest/ folders."""
        try:
            import os
            import shutil

            # Get source experiment directory
            experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)

            # Get best and latest directories
            best_dir = self.path_manager.get_best_folder_path()
            latest_dir = self.path_manager.get_latest_folder_path()

            # Files/directories to copy (evaluation-specific)
            items_to_copy = ["evaluation", "plots"]

            for item_name in items_to_copy:
                source_path = os.path.join(experiment_dir, item_name)
                if os.path.exists(source_path):
                    # Copy to best/
                    best_dest = os.path.join(best_dir, item_name)
                    if os.path.exists(best_dest):
                        if os.path.isdir(best_dest):
                            shutil.rmtree(best_dest)
                        else:
                            os.remove(best_dest)

                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, best_dest)
                    else:
                        shutil.copy2(source_path, best_dest)

                    # Copy to latest/
                    latest_dest = os.path.join(latest_dir, item_name)
                    if os.path.exists(latest_dest):
                        if os.path.isdir(latest_dest):
                            shutil.rmtree(latest_dest)
                        else:
                            os.remove(latest_dest)

                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, latest_dest)
                    else:
                        shutil.copy2(source_path, latest_dest)

            logger.debug("Copied evaluation results to best/ and latest/ folders")

        except Exception as e:
            logger.warning(f"Could not copy evaluation results to symlink folders: {e}")

    def run_analysis_script(self):
        """Run custom analysis script if enabled."""
        if not hasattr(self.config_instance.analysis, "run_analysis_script"):
            return

        script_config = self.config_instance.analysis.run_analysis_script
        if not getattr(script_config, "enabled", False):
            return

        # Use custom path if provided, otherwise default to 'analysis.py'
        script_path = getattr(script_config, "script_path", "analysis.py")

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
