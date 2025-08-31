"""
Storage management utilities for experiment cleanup and optimization.

This module provides automatic cleanup policies for experiment directories
to manage storage efficiently while preserving important experiments.
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class StorageManager:
    """
    Manages storage policies for experiments including cleanup and optimization.
    """

    def __init__(self, config: Any, path_manager):
        """
        Initialize storage manager.

        Args:
            config: Configuration object with storage settings
            path_manager: PathManager instance
        """
        self.config = config
        self.path_manager = path_manager
        self.storage_config = getattr(config, "storage", {})
        self.cleanup_policy = self.storage_config.get("cleanup_policy", {})
        self.model_saving = self.storage_config.get("model_saving", {})

    def get_experiments_dir(self) -> Path:
        """Get experiments directory path."""
        project_root = self.path_manager._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )
        task_type = getattr(self.config.model, "task_type", "classification")
        return project_root / "outputs" / project_name / "experiments" / correction_dir / task_type

    def list_experiments(self) -> list[dict[str, Any]]:
        """
        List all experiments with their metadata.

        Returns:
            List of experiment info dictionaries
        """
        experiments_dir = self.get_experiments_dir()
        experiments = []

        if not experiments_dir.exists():
            return experiments

        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                # Skip symlinks (best, latest)
                if exp_dir.is_symlink():
                    continue

                exp_info = {
                    "name": exp_dir.name,
                    "path": str(exp_dir),
                    "created": datetime.fromtimestamp(exp_dir.stat().st_ctime),
                    "size_mb": self._calculate_dir_size(exp_dir) / (1024 * 1024),
                }

                # Load metadata if available
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        exp_info["metadata"] = json.load(f)

                experiments.append(exp_info)

        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x["created"], reverse=True)
        return experiments

    def _calculate_dir_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        for path in directory.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size

    def get_protected_experiments(self) -> set:
        """
        Get set of experiment names that should not be deleted.

        Returns:
            Set of protected experiment names
        """
        protected = set()
        experiments_dir = self.get_experiments_dir()

        if not experiments_dir.exists():
            return protected

        # Protect experiments linked in best/
        best_dir = experiments_dir / "best"
        if best_dir.exists():
            for symlink in best_dir.iterdir():
                if symlink.is_symlink():
                    target = symlink.resolve()
                    if target.exists():
                        protected.add(target.name)

        # Protect latest
        latest_link = experiments_dir / "latest"
        if latest_link.exists() and latest_link.is_symlink():
            target = latest_link.resolve()
            if target.exists():
                protected.add(target.name)

        return protected

    def should_cleanup_experiment(self, exp_info: dict[str, Any]) -> bool:
        """
        Determine if an experiment should be cleaned up based on policy.

        Args:
            exp_info: Experiment information dictionary

        Returns:
            bool: True if experiment should be deleted
        """
        # Never delete protected experiments
        protected = self.get_protected_experiments()
        if exp_info["name"] in protected:
            return False

        # Check age policy
        keep_days = self.cleanup_policy.get("keep_days", 30)
        if keep_days > 0:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            if exp_info["created"] < cutoff_date:
                return True

        return False

    def cleanup_experiments(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Clean up experiments based on storage policy.

        Args:
            dry_run: If True, only simulate cleanup without deleting

        Returns:
            Dictionary with cleanup results
        """
        if not self.cleanup_policy.get("auto_cleanup", False):
            logger.debug("Auto cleanup is disabled")
            return {"cleaned": 0, "protected": 0, "total_size_freed": 0}

        experiments = self.list_experiments()
        protected = self.get_protected_experiments()

        # Apply keep_last_n policy
        keep_last_n = self.cleanup_policy.get("keep_last_n", 10)
        if keep_last_n > 0 and len(experiments) > keep_last_n:
            # Keep the N most recent, mark rest for cleanup (unless protected)
            candidates_for_cleanup = experiments[keep_last_n:]
        else:
            # Apply age-based cleanup
            candidates_for_cleanup = [
                exp for exp in experiments if self.should_cleanup_experiment(exp)
            ]

        cleaned_count = 0
        total_size_freed = 0

        for exp_info in candidates_for_cleanup:
            if exp_info["name"] in protected:
                logger.info(f"Skipping protected experiment: {exp_info['name']}")
                continue

            if dry_run:
                logger.info(
                    f"Would delete experiment: {exp_info['name']} ({exp_info['size_mb']:.1f} MB)"
                )
            else:
                logger.info(
                    f"Deleting experiment: {exp_info['name']} ({exp_info['size_mb']:.1f} MB)"
                )
                shutil.rmtree(exp_info["path"])
                cleaned_count += 1

            total_size_freed += exp_info["size_mb"]

        results = {
            "cleaned": cleaned_count,
            "protected": len(protected),
            "total_size_freed": total_size_freed,
            "experiments_remaining": len(experiments) - cleaned_count,
        }

        if not dry_run and cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} experiments, freed {total_size_freed:.1f} MB"
            )

        return results

    def should_save_model(self, model_improved: bool) -> bool:
        """
        Determine if model should be saved based on policy.

        Args:
            model_improved: Whether the model achieved better validation performance

        Returns:
            bool: True if model should be saved
        """
        save_only_if_best = self.model_saving.get("save_model_only_if_best", True)

        if save_only_if_best:
            return model_improved
        else:
            return True  # Always save if policy allows

    def save_outputs(
        self,
        pipeline,
        training_visuals=False,
        evaluation_visuals=False,
        metadata=False,
        symlinks=False,
        result_type=None,
        ):
        """
        Unified method to save various pipeline outputs.

        Args:
            pipeline: PipelineManager instance with all the data
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
                    "epochs_run": len(pipeline.history.history.get("loss", [])),
                    "best_epoch": getattr(pipeline, "best_epoch", None),
                    "best_val_loss": min(
                        pipeline.history.history.get("val_loss", [float("inf")])
                    ),
                    "model_improved": getattr(pipeline, "model_improved", False),
                },
            }

            # Only add evaluation data if it exists
            if hasattr(pipeline, "last_mae"):
                training_data["evaluation"] = {
                    "mae": pipeline.last_mae,
                    "r2": getattr(pipeline, "last_r2", None),
                    "pearson": getattr(pipeline, "last_pearson", None),
                }

            pipeline.path_manager.save_experiment_metadata(
                pipeline.experiment_name, training_data
            )

        # Save model and update folders if requested
        if symlinks:
            # For training+evaluation pipeline, always save the model
            # For standalone, respect the storage policy
            model_path = pipeline.path_manager.get_experiment_model_path(
                pipeline.experiment_name
            )
            import os
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            pipeline.model.save(model_path)

            # Update best folder if this is a new best model
            if pipeline.model_improved:
                pipeline.path_manager.update_best_folder(pipeline.experiment_name)

            # Always update latest folder
            pipeline.path_manager.update_latest_folder(pipeline.experiment_name)

        # Save visualizations if requested and available
        if (
            training_visuals or evaluation_visuals
        ) and pipeline.visualizer_class is not None:
            # Determine what data to pass based on visualization type
            test_data = pipeline.test_data if evaluation_visuals else None
            test_labels = pipeline.test_labels if evaluation_visuals else None
            shap_values = (
                getattr(pipeline, "squeezed_shap_values", None)
                if evaluation_visuals
                else None
            )
            shap_data = (
                getattr(pipeline, "squeezed_test_data", None)
                if evaluation_visuals
                else None
            )
            adata = getattr(pipeline, "adata", None) if evaluation_visuals else None
            adata_corrected = (
                getattr(pipeline, "adata_corrected", None) if evaluation_visuals else None
            )

            # Create visualizer
            visualizer = pipeline.visualizer_class(
                pipeline.config_instance,
                pipeline.model,
                pipeline.history,
                test_data,
                test_labels,
                pipeline.label_encoder,
                shap_values,
                shap_data,
                adata,
                adata_corrected,
                pipeline.path_manager,
            )

            # Set evaluation context if result_type provided (for symlinks)
            if result_type:
                visualizer.set_evaluation_context(result_type)
            elif evaluation_visuals or training_visuals:
                visualizer.set_evaluation_context(pipeline.experiment_name)

            # Generate appropriate visualizations
            if training_visuals:
                visualizer._visualize_training_history()

            if evaluation_visuals:
                # Set output directory for evaluation plots
                plots_dir = pipeline.path_manager.get_experiment_plots_dir(
                    pipeline.experiment_name
                )
                visualizer.visual_tools.output_dir = plots_dir
                visualizer.run()

        elif (training_visuals or evaluation_visuals) and pipeline.visualizer_class is None:
            logger.warning("Visualizer class not available, skipping visualizations")

