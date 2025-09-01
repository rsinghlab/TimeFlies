"""
Display Manager for TimeFlies Pipeline

Handles all printing, formatting, and console output for the pipeline.
Extracted from PipelineManager to reduce complexity and improve maintainability.
"""

import logging

logger = logging.getLogger(__name__)


class DisplayManager:
    """Manages all display and printing functionality for TimeFlies pipelines."""

    def __init__(self, config=None):
        self.config = config

    def print_timeflies_header(self, training_mode=True):
        """Print TimeFlies header."""
        if training_mode:
            self._print_header("TIMEFLIES TRAINING & HOLDOUT EVALUATION")
        else:
            self._print_header("TIMEFLIES EVALUATION")

    def print_project_and_dataset_overview(self, config, pipeline=None, display_format=None):
        """Print project information and dataset overview."""
        print(f"Project: {config.project.replace('_', ' ').title()}")
        
        # Show experiment name with display format if provided
        if pipeline and display_format:
            print(f"Experiment: {pipeline.experiment_name} ({display_format})")
        else:
            print(
                f"Experiment: {getattr(config, 'experiment_name', 'Unknown')} ({getattr(config, 'run_name', 'Unknown')})"
            )

        # Hardware info
        hardware_type = getattr(config.hardware, "processor", "Unknown")
        if hardware_type == "GPU":
            print("GPU: Available")
        elif hardware_type == "M":
            print("GPU: Not available (using CPU)")
        else:
            print(f"GPU: {hardware_type}")

        # Task and batch correction info
        task_type = getattr(config.model, "task_type", "classification")
        print(f"Task Type: {task_type.title()}")

        batch_enabled = getattr(config.data.batch_correction, "enabled", False)
        print(f"Batch Correction: {'Enabled' if batch_enabled else 'Disabled'}")

        # Gene filtering warning - check for reference data files
        self._check_gene_filtering_status(config)

    def print_training_and_evaluation_data(
        self, train_data, eval_data, config, train_subset=None, eval_subset=None
        ):
        """Print comprehensive overview of both training and evaluation datasets."""
        self._print_header("PREPROCESSED DATA OVERVIEW")

        # Training data info
        print("Training Data:")
        print(f"  └─ Samples:           {train_data.shape[0]:,}")
        print(
            f"  └─ Features:          {' x '.join(map(str, train_data.shape[1:]))} (reshaped)"
        )

        if hasattr(train_data, "min") and hasattr(train_data, "max"):
            print(
                f"  └─ Data Range:        [{train_data.min():.3f}, {train_data.max():.3f}]"
            )
            print(
                f"  └─ Mean ± Std:        {train_data.mean():.3f} ± {train_data.std():.3f}"
            )

        # Distribution info for training
        if train_subset is not None:
            target_var = getattr(config.data, "target_variable", "age")
            self._print_obs_distributions(train_subset, [target_var, "sex"])

        print()

        # Evaluation data info
        print("Evaluation Data:")
        print(f"  └─ Samples:           {eval_data.shape[0]:,}")
        print(
            f"  └─ Features:          {' x '.join(map(str, eval_data.shape[1:]))} (reshaped)"
        )

        if hasattr(eval_data, "min") and hasattr(eval_data, "max"):
            print(
                f"  └─ Data Range:        [{eval_data.min():.3f}, {eval_data.max():.3f}]"
            )
            print(
                f"  └─ Mean ± Std:        {eval_data.mean():.3f} ± {eval_data.std():.3f}"
            )

        # Distribution info for evaluation
        if eval_subset is not None:
            target_var = getattr(config.data, "target_variable", "age")
            self._print_obs_distributions(eval_subset, [target_var, "sex"])

        # Split configuration
        split_method = getattr(config.data.split, "method", "unknown")
        print()
        print("Split Configuration:")
        print(f"  └─ Split Method:      {split_method.title()}")

    def print_evaluation_data(self, eval_data, config, eval_subset=None):
        """Print evaluation dataset overview."""
        self._print_header("EVALUATION DATASET OVERVIEW")

        if eval_data is not None:
            print("Evaluation Data:")
            print(f"  └─ Samples:           {eval_data.shape[0]:,}")
            print(
                f"  └─ Features:          {' x '.join(map(str, eval_data.shape[1:]))} (reshaped)"
            )

            if hasattr(eval_data, "min") and hasattr(eval_data, "max"):
                print(
                    f"  └─ Data Range:        [{eval_data.min():.3f}, {eval_data.max():.3f}]"
                )
                print(
                    f"  └─ Mean ± Std:        {eval_data.mean():.3f} ± {eval_data.std():.3f}"
                )
        else:
            print("Evaluation Data: None")

        # Distribution info from eval_subset AnnData
        if eval_subset is not None:
            target_var = getattr(config.data, "target_variable", "age")
            self._print_obs_distributions(eval_subset, [target_var, "sex"])

        # Split configuration
        split_method = getattr(config.data.split, "method", "unknown")
        print()
        print("Split Configuration:")
        print(f"  └─ Split Method:      {split_method.title()}")

    def display_model_architecture(self, model, config):
        """Display model architecture and training configuration."""
        self._print_header("MODEL ARCHITECTURE")

        model_type = getattr(config.data, "model", "Unknown")
        print(f"Model Type: {model_type}")
        print()

        # Training configuration
        training_config = getattr(config.model, "training", None)
        if training_config:
            print("Training Configuration:")
            print(
                "  └─ Optimizer:              Adam"
            )  # Hardcoded in current implementation
            print(
                f"  └─ Learning Rate:          {getattr(training_config, 'learning_rate', 'Unknown')}"
            )
            print(
                f"  └─ Batch Size:             {getattr(training_config, 'batch_size', 'Unknown')}"
            )
            print(
                f"  └─ Max Epochs:             {getattr(training_config, 'epochs', 'Unknown')}"
            )
            print(
                f"  └─ Validation Split:      {getattr(training_config, 'validation_split', 'Unknown')}"
            )
            print(
                f"  └─ Early Stopping Patience: {getattr(training_config, 'early_stopping_patience', 'Unknown')}"
            )
            print()

        # Model summary
        if hasattr(model, "summary"):
            model.summary()

    def get_previous_best_loss_message(self, path_manager):
        """Get formatted message about previous best validation loss."""
        try:
            prev_best = path_manager.get_previous_best_validation_loss()
            if prev_best is not None:
                return f"Previous best validation loss: {prev_best:.3f}"
            else:
                return "No previous validation loss found"
        except Exception as e:
            logger.debug(f"Could not get previous best loss: {e}")
            return "Could not determine previous validation loss"

    def print_training_progress_header(self, previous_best_msg):
        """Print training progress header with previous best loss info."""
        self._print_header(f"TRAINING PROGRESS ({previous_best_msg})")
    
    def print_training_results(self, history, original_previous_best_loss, 
                              experiment_name, improvement_status):
        """Print training results section."""
        self._print_header("TRAINING RESULTS")
        current_best_loss = None
        
        if history:
            val_losses = history.history.get("val_loss", [])
            if val_losses:
                best_epoch = val_losses.index(min(val_losses)) + 1
                current_best_loss = min(val_losses)
                print(f"  └─ Best Epoch:                {best_epoch}")
                print(f"  └─ Best Val Loss (This Run):  {current_best_loss:.4f}")

        # Show previous best and change
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

        print(f"  └─ Model Saved To:            {experiment_name}")
        print(f"  └─ Status:                    {improvement_status}")

    def print_model_comparison_header(self):
        """Print training results header."""
        self._print_header("MODEL COMPARISON (Holdout Evaluation)")

    def print_timing_summary(self, preprocessing_duration=0, training_duration=0, 
                            evaluation_duration=0):
        """Print timing summary for any pipeline mode."""
        self._print_header("TIMING SUMMARY")
        
        # Show only the durations that are non-zero
        if preprocessing_duration > 0:
            print(f"Preprocessing:        {self._format_duration(preprocessing_duration)}")
        if training_duration > 0:
            print(f"Training:             {self._format_duration(training_duration)}")
        if evaluation_duration > 0:
            print(f"Evaluation:           {self._format_duration(evaluation_duration)}")
        
        total_duration = preprocessing_duration + training_duration + evaluation_duration
        if total_duration > 0:
            print(f"Total Duration:       {self._format_duration(total_duration)}")

    def _print_header(self, title: str, width: int = 60):
        """Print a formatted header with title."""
        print()
        print(title.upper())
        print("=" * width)

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

    def _check_gene_filtering_status(self, config):
        """Check if gene filtering reference data exists and display warning if missing."""
        from pathlib import Path
        
        project = getattr(config, "project", "unknown")
        
        # Get reference data filenames from config or use defaults
        autosomal_file = getattr(config.data, "autosomal_genes_file", "autosomal_genes.csv")
        sex_file = getattr(config.data, "sex_genes_file", "sex_genes.csv")
        
        # Check if reference data files exist
        autosomal_path = Path(f"data/{project}/reference_data/{autosomal_file}")
        sex_path = Path(f"data/{project}/reference_data/{sex_file}")
        
        if not autosomal_path.exists() and not sex_path.exists():
            print("⚠ Gene filtering disabled (no reference data found)")

    def _print_obs_distributions(self, adata, columns: list[str]):
        """Print distribution information for specified columns."""
        for column in columns:
            if column in adata.obs.columns:
                print(f"  └─ {column.title()} Distribution:")
                value_counts = adata.obs[column].value_counts()
                total = len(adata.obs)
                for value, count in value_counts.items():
                    percentage = (count / total) * 100
                    print(
                        f"      └─ {value:10} : {count:6,} samples ({percentage:5.1f}%)"
                    )
    