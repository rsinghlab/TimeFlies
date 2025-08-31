"""
Display Manager for TimeFlies Pipeline

Handles all printing, formatting, and console output for the pipeline.
Extracted from PipelineManager to reduce complexity and improve maintainability.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DisplayManager:
    """Manages all display and printing functionality for TimeFlies pipelines."""
    
    def __init__(self, config=None):
        self.config = config
    
    def print_header(self, title: str, width: int = 60):
        """Print a formatted header with title."""
        print("=" * width)
        print(title.upper())
        print("=" * width)
    
    def print_project_and_dataset_overview(self, config, adata, adata_eval=None):
        """Print project information and dataset overview."""
        print(f"Project: {config.project.replace('_', ' ').title()}")
        print(f"Experiment: {getattr(config, 'experiment_name', 'Unknown')} ({getattr(config, 'run_name', 'Unknown')})")
        
        # Hardware info
        hardware_type = getattr(config.hardware, 'processor', 'Unknown')
        if hardware_type == "GPU":
            print("GPU: Available")
        elif hardware_type == "M":
            print("GPU: Not available (using CPU)")
        else:
            print(f"GPU: {hardware_type}")
            
        # Task and batch correction info
        task_type = getattr(config.model, 'task_type', 'classification')
        print(f"Task Type: {task_type.title()}")
        
        batch_enabled = getattr(config.data.batch_correction, 'enabled', False)
        print(f"Batch Correction: {'Enabled' if batch_enabled else 'Disabled'}")
        
        # Gene filtering warning
        if hasattr(config.data, 'gene_filtering_disabled') and config.data.gene_filtering_disabled:
            print("⚠ Gene filtering disabled (no reference data found)")
    
    def print_obs_distributions(self, adata, columns: List[str]):
        """Print distribution information for specified columns."""
        for column in columns:
            if column in adata.obs.columns:
                print(f"  └─ {column.title()} Distribution:")
                value_counts = adata.obs[column].value_counts()
                total = len(adata.obs)
                for value, count in value_counts.items():
                    percentage = (count / total) * 100
                    print(f"      └─ {value:10} : {count:6,} samples ({percentage:5.1f}%)")
    
    def print_training_and_evaluation_data(self, train_data, eval_data, config, train_subset=None, eval_subset=None):
        """Print comprehensive overview of both training and evaluation datasets."""
        self.print_header("PREPROCESSED DATA OVERVIEW")
        
        # Training data info
        print("Training Data:")
        print(f"  └─ Samples:           {train_data.shape[0]:,}")
        print(f"  └─ Features:          {' x '.join(map(str, train_data.shape[1:]))} (reshaped)")
        
        if hasattr(train_data, 'min') and hasattr(train_data, 'max'):
            print(f"  └─ Data Range:        [{train_data.min():.3f}, {train_data.max():.3f}]")
            print(f"  └─ Mean ± Std:        {train_data.mean():.3f} ± {train_data.std():.3f}")
        
        # Distribution info for training
        if train_subset is not None:
            target_var = getattr(config.data, 'target_variable', 'age')
            self.print_obs_distributions(train_subset, [target_var, 'sex'])
        
        print()
        
        # Evaluation data info
        print("Evaluation Data:")
        print(f"  └─ Samples:           {eval_data.shape[0]:,}")
        print(f"  └─ Features:          {' x '.join(map(str, eval_data.shape[1:]))} (reshaped)")
        
        if hasattr(eval_data, 'min') and hasattr(eval_data, 'max'):
            print(f"  └─ Data Range:        [{eval_data.min():.3f}, {eval_data.max():.3f}]")
            print(f"  └─ Mean ± Std:        {eval_data.mean():.3f} ± {eval_data.std():.3f}")
        
        # Distribution info for evaluation
        if eval_subset is not None:
            target_var = getattr(config.data, 'target_variable', 'age')
            self.print_obs_distributions(eval_subset, [target_var, 'sex'])
        
        # Split configuration
        split_method = getattr(config.data.split, 'method', 'unknown')
        print()
        print("Split Configuration:")
        print(f"  └─ Split Method:      {split_method.title()}")
    
    def print_evaluation_data(self, eval_data, config, eval_subset=None):
        """Print evaluation dataset overview."""
        self.print_header("EVALUATION DATASET OVERVIEW")
        
        if eval_data is not None:
            print("Evaluation Data:")
            print(f"  └─ Samples:           {eval_data.shape[0]:,}")
            print(f"  └─ Features:          {' x '.join(map(str, eval_data.shape[1:]))} (reshaped)")
            
            if hasattr(eval_data, 'min') and hasattr(eval_data, 'max'):
                print(f"  └─ Data Range:        [{eval_data.min():.3f}, {eval_data.max():.3f}]")
                print(f"  └─ Mean ± Std:        {eval_data.mean():.3f} ± {eval_data.std():.3f}")
        else:
            print("Evaluation Data: None")
        
        # Distribution info from eval_subset AnnData
        if eval_subset is not None:
            target_var = getattr(config.data, 'target_variable', 'age')
            self.print_obs_distributions(eval_subset, [target_var, 'sex'])
        
        # Split configuration
        split_method = getattr(config.data.split, 'method', 'unknown')
        print()
        print("Split Configuration:")
        print(f"  └─ Split Method:      {split_method.title()}")
    
    def display_model_architecture(self, model, config):
        """Display model architecture and training configuration."""
        self.print_header("MODEL ARCHITECTURE")
        print("-" * 60)
        print()
        
        model_type = getattr(config.data, 'model', 'Unknown')
        print(f"Model Type: {model_type}")
        print()
        
        # Training configuration
        training_config = getattr(config.model, 'training', None)
        if training_config:
            print("Training Configuration:")
            print(f"  └─ Optimizer:              Adam")  # Hardcoded in current implementation
            print(f"  └─ Learning Rate:          {getattr(training_config, 'learning_rate', 'Unknown')}")
            print(f"  └─ Batch Size:             {getattr(training_config, 'batch_size', 'Unknown')}")
            print(f"  └─ Max Epochs:             {getattr(training_config, 'epochs', 'Unknown')}")
            print(f"  └─ Validation Split:      {getattr(training_config, 'validation_split', 'Unknown')}")
            print(f"  └─ Early Stopping Patience: {getattr(training_config, 'early_stopping_patience', 'Unknown')}")
            print()
        
        # Model summary
        if hasattr(model, 'summary'):
            model.summary()
    
    def show_processed_eval_data_preview(self, adata_eval):
        """Show a preview of the processed evaluation data."""
        if adata_eval is None:
            print("⚠️  No evaluation data available")
            return
            
        print("\nEVALUATION DATA PREVIEW")
        print("-" * 40)
        print(f"Shape: {adata_eval.shape}")
        
        if hasattr(adata_eval, 'obs'):
            print(f"Observations: {adata_eval.obs.shape[0]:,}")
            print(f"Features: {adata_eval.shape[1]:,}")
    
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
    
    def print_final_timing_summary(self, evaluation_duration: float, preprocessing_duration: float = 0, mode: str = "evaluation"):
        """Print final timing summary."""
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            else:
                hours = seconds / 3600
                return f"{hours:.1f}h"
        
        print()
        print("TIMING SUMMARY")
        print("-" * 60)
        
        if mode == "evaluation":
            # Standalone evaluation mode
            if preprocessing_duration > 0:
                print(f"  └─ Preprocessing:         {format_duration(preprocessing_duration)}")
            print(f"  └─ Evaluation:            {format_duration(evaluation_duration)}")
            
            total_duration = preprocessing_duration + evaluation_duration
            print(f"  └─ Total Duration:        {format_duration(total_duration)}")
        else:
            # Combined or other modes - just show evaluation
            print(f"  └─ Evaluation Duration:   {format_duration(evaluation_duration)}")