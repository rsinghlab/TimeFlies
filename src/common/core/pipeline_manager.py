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
        self.model_manager = ModelManager(self.config_instance, self.path_manager)
        self.mode = mode

        # Set experiment name based on mode
        if mode == "training":
            # Generate new experiment for training
            self.experiment_name = self.path_manager.generate_experiment_name()
        else:
            # Reuse best experiment for standalone evaluation
            try:
                self.experiment_name = self.path_manager.get_best_experiment_name()
                if not self.experiment_name:
                    raise RuntimeError("No best experiment name found")
            except (FileNotFoundError, RuntimeError) as e:
                # If no best experiment exists, this will fail with a clear error
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

    def run_training(self, skip_setup=False) -> dict[str, Any]:
        """
        Run the complete training pipeline: setup + preprocessing + model training.

        Args:
            skip_setup: Skip pipeline setup if already done (for combined pipeline)

        Returns:
            Dict containing training results and paths
        """

        if not skip_setup:
            self.display_manager.print_header("🔥 TRAINING PIPELINE")
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
            print(f"Could not display training and evaluation data: {e}")

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
        self.display_manager.print_header("MODEL ARCHITECTURE")
        self.display_manager.display_model_architecture(
            self.model, self.config_instance
        )

        # Model training
        print("\n")
        print(
            f"TRAINING PROGRESS ({self.model_manager.get_previous_best_loss_message(self.experiment_name)})"
        )
        print("=" * 60)

        # Store original previous best loss for comparison (before training updates it)
        self._original_previous_best_loss = (
            self.model_manager.get_previous_best_validation_loss(self.experiment_name)
        )

        self.history, self.model, self.model_improved = self.model_manager.train_model(
            self.model_builder, self.model
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
            self.storage_manager.save_outputs(
                self, training_visuals=True
            )  # Save to experiment directory

        # Save outputs and create symlinks after training
        self.storage_manager.save_outputs(self, metadata=True, symlinks=True)

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
        original_previous_best_loss = getattr(
            self, "_original_previous_best_loss", None
        )

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
                print(f"Could not display evaluation data: {e}")

            # Display model architecture (similar to training pipeline)
            print("\n")
            self.display_manager.print_header("MODEL ARCHITECTURE")
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
            self.display_manager.print_final_timing_summary(
                evaluation_duration,
                preprocessing_duration=preprocessing_duration,
                mode="evaluation",
            )
        # In combined pipeline, timing is handled by run_pipeline()

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

        # Update best/latest folders to include evaluation results
        if evaluation_results and self.experiment_name:
            # Only update best if model actually improved (not just any training run)
            if hasattr(self, "model_improved") and self.model_improved:
                self.path_manager.update_best_folder(self.experiment_name)
            # Always update latest to reflect most recent activity
            self.path_manager.update_latest_folder(self.experiment_name)

        # Display timing summary for combined pipeline
        print()
        self.display_manager.print_header("TIMING SUMMARY")

        # Get stored durations from different phases
        preprocessing_duration = getattr(self, "_preprocessing_duration", 0)
        training_duration = getattr(self, "_stored_training_duration", 0)
        evaluation_duration = evaluation_results.get("evaluation_duration", 0)

        print(f"Preprocessing:        {self._format_duration(preprocessing_duration)}")
        print(f"Training:             {self._format_duration(training_duration)}")
        print(f"Evaluation:           {self._format_duration(evaluation_duration)}")

        total_duration = (
            preprocessing_duration + training_duration + evaluation_duration
        )
        print(f"Total Duration:       {self._format_duration(total_duration)}")

        # Memory cleanup removed - causes problems
        self._in_combined_pipeline = False

        return pipeline_results

    def _format_duration(self, seconds):
        """Format duration for display."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

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

        # Create interpreter with specific output directory
        interpreter = self.interpreter_class(
            config=self.config_instance,
            model=self.model,
            test_data=self.test_data,
            test_labels=self.test_labels,
            label_encoder=self.label_encoder,
            path_manager=self.path_manager,
            output_dir=evaluation_output_dir,
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
