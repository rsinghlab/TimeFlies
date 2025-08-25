"""
Model Queue Manager for automated sequential model training.

This module provides functionality to train multiple models sequentially
with different hyperparameters and generate comprehensive comparison reports.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from common.cli.commands import evaluate_command, train_command
from common.core.config_manager import get_config_manager
from common.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelQueueManager:
    """Manages sequential training of multiple models with progress tracking."""

    def __init__(self, queue_config_path: str):
        """
        Initialize the Model Queue Manager.

        Args:
            queue_config_path: Path to the queue configuration YAML file
        """
        self.queue_config_path = Path(queue_config_path)
        self.load_queue_config()
        self.results = []
        self.checkpoint_file = Path("outputs/.queue_checkpoint.json")
        self.start_time = None
        self.completed_models = []

    def load_queue_config(self):
        """Load and validate the queue configuration."""
        if not self.queue_config_path.exists():
            raise FileNotFoundError(f"Queue config not found: {self.queue_config_path}")

        with open(self.queue_config_path) as f:
            self.config = yaml.safe_load(f)

        # Validate required fields
        if "model_queue" not in self.config:
            raise ValueError("Queue config must contain 'model_queue' section")

        self.queue_settings = self.config.get("queue_settings", {})
        self.global_settings = self.config.get("global_settings", {})
        self.model_queue = self.config["model_queue"]

        logger.info(f"Loaded queue with {len(self.model_queue)} models")

    def load_checkpoint(self) -> int | None:
        """
        Load checkpoint to resume interrupted queue.

        Returns:
            Index of last completed model, or None if no checkpoint
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
                self.completed_models = checkpoint.get("completed_models", [])
                self.results = checkpoint.get("results", [])
                return checkpoint.get("last_completed_index", -1)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return None

    def save_checkpoint(self, model_index: int):
        """Save progress checkpoint."""
        if not self.queue_settings.get("save_checkpoints", True):
            return

        checkpoint = {
            "last_completed_index": model_index,
            "completed_models": self.completed_models,
            "results": self.results,
            "timestamp": datetime.now().isoformat(),
        }

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def prepare_model_config(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare configuration for a single model.

        Args:
            model_config: Model-specific configuration

        Returns:
            Complete configuration with global settings merged
        """
        # Start with global settings
        config = self._deep_merge_dict(self.global_settings.copy(), {})

        # Add model-specific settings
        config["model"] = model_config.get("model_type", "CNN")
        config["experiment_name"] = model_config.get("name", "unnamed_model")

        # Apply any per-model configuration overrides
        config_overrides = model_config.get("config_overrides", {})
        if config_overrides:
            config = self._deep_merge_dict(config, config_overrides)
            logger.info(
                f"Applied config overrides for {model_config.get('name', 'unnamed')}"
            )

        # Merge hyperparameters
        hyperparams = model_config.get("hyperparameters", {})

        # Apply hyperparameters based on model type
        model_type = model_config.get("model_type", "CNN").lower()

        if model_type in ["cnn", "mlp"]:
            if "epochs" in hyperparams:
                config["epochs"] = hyperparams["epochs"]
            if "batch_size" in hyperparams:
                config["batch_size"] = hyperparams["batch_size"]
            if "learning_rate" in hyperparams:
                config["learning_rate"] = hyperparams["learning_rate"]

        return config, hyperparams

    def _deep_merge_dict(self, base_dict: dict, override_dict: dict) -> dict:
        """
        Deep merge two dictionaries, with override_dict taking precedence.

        Args:
            base_dict: Base dictionary
            override_dict: Dictionary with override values

        Returns:
            Merged dictionary
        """
        result = base_dict.copy()

        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value

        return result

    def train_single_model(
        self, model_index: int, model_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Train a single model from the queue.

        Args:
            model_index: Index in the queue
            model_config: Model configuration

        Returns:
            Dictionary with training results and metrics
        """
        model_name = model_config.get("name", f"model_{model_index}")
        model_type = model_config.get("model_type", "CNN")
        description = model_config.get("description", "")

        print("\n" + "=" * 60)
        print(f"[{model_index + 1}/{len(self.model_queue)}] Training: {model_name}")
        print(f"Model Type: {model_type}")
        if description:
            print(f"Description: {description}")
        print("=" * 60)

        start_time = time.time()

        try:
            # Prepare configuration
            config, hyperparams = self.prepare_model_config(model_config)

            # Create mock args object for CLI commands
            class Args:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            # Set up training arguments with actual configuration support
            train_args = Args(
                verbose=False,
                model=model_type,
                tissue=config.get("data", {}).get("tissue", "head"),
                target=config.get("data", {}).get("target_variable", "age"),
                project=config.get("project"),
                batch_corrected=config.get("data", {})
                .get("batch_correction", {})
                .get("enabled", False),
                with_eda=config.get("with_eda", False),
                with_analysis=config.get("with_analysis", True),
            )

            # Store full config for this model to be used during training
            self._current_model_config = config

            # Run training if configured
            should_train = config.get("with_training", True)  # New setting for training
            if should_train:
                print(f"\n[INFO] Starting training for {model_name}...")
                train_command(train_args, config)
            else:
                print(
                    f"\n[INFO] Skipping training for {model_name} (with_training: false)"
                )

            # Run evaluation if configured (separate from with_analysis which is for scripts)
            should_evaluate = config.get(
                "with_evaluation", True
            )  # New setting for evaluation
            if should_evaluate:
                eval_args = Args(
                    verbose=False,
                    model=model_type,
                    tissue=config.get("data", {}).get("tissue", "head"),
                    target=config.get("data", {}).get("target_variable", "age"),
                    project=config.get("project"),
                    batch_corrected=config.get("data", {})
                    .get("batch_correction", {})
                    .get("enabled", False),
                    with_eda=config.get("with_eda", False),
                    with_analysis=config.get("with_analysis", True),
                    interpret=config.get("interpret", True),
                    visualize=config.get("visualize", True),
                )

                print(f"\n[INFO] Starting evaluation for {model_name}...")
                evaluate_command(eval_args, config)

            training_time = time.time() - start_time

            # Collect results
            result = {
                "name": model_name,
                "model_type": model_type,
                "description": description,
                "hyperparameters": hyperparams,
                "training_time": training_time,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Try to load metrics from the latest evaluation
            metrics_path = self._find_latest_metrics(model_type, config)
            if metrics_path and metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    result["metrics"] = metrics

            print(f"\n[OK] Model {model_name} completed in {training_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            return {
                "name": model_name,
                "model_type": model_type,
                "description": description,
                "hyperparameters": model_config.get("hyperparameters", {}),
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _find_latest_metrics(self, model_type: str, config: dict) -> Path | None:
        """Find the latest metrics file for a model."""
        project = config.get("project", "fruitfly_aging")

        # Build expected path
        base_path = Path("outputs") / project / "experiments"
        if config.get("batch_corrected"):
            base_path = base_path / "batch_corrected"
        else:
            base_path = base_path / "uncorrected"

        latest_link = base_path / "latest"

        if latest_link.exists() and latest_link.is_symlink():
            metrics_file = latest_link / "evaluation" / "metrics.json"
            if metrics_file.exists():
                return metrics_file

        return None

    def run_queue(self, resume: bool = True):
        """
        Run the complete model queue.

        Args:
            resume: Whether to resume from checkpoint if available
        """
        self.start_time = time.time()

        # Check for checkpoint
        start_index = 0
        if resume:
            checkpoint_index = self.load_checkpoint()
            if checkpoint_index is not None:
                start_index = checkpoint_index + 1
                print(
                    f"\n[INFO] Resuming from checkpoint (completed {checkpoint_index + 1} models)"
                )

        total_models = len(self.model_queue)

        print("\n" + "=" * 60)
        print(f"STARTING MODEL QUEUE: {total_models} models to train")
        print(f"Queue name: {self.queue_settings.get('name', 'unnamed')}")
        print("=" * 60)

        # Train each model
        for i in range(start_index, total_models):
            model_config = self.model_queue[i]

            # Show progress
            remaining = total_models - i
            print(f"\n[PROGRESS] {i} completed, {remaining} remaining")

            # Train model
            result = self.train_single_model(i, model_config)
            self.results.append(result)
            self.completed_models.append(model_config.get("name", f"model_{i}"))

            # Save checkpoint
            self.save_checkpoint(i)

            # Show summary so far
            if i > 0:
                self._print_progress_summary()

        # Generate final summary report
        if self.queue_settings.get("generate_summary", True):
            self.generate_summary_report()

        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(
            f"[OK] QUEUE COMPLETED: {total_models} models in {total_time / 60:.2f} minutes"
        )
        print("=" * 60)

        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def _print_progress_summary(self):
        """Print a quick summary of progress so far."""
        completed = len([r for r in self.results if r.get("status") == "completed"])
        failed = len([r for r in self.results if r.get("status") == "failed"])

        print("\n" + "-" * 40)
        print(f"Progress: {completed} completed, {failed} failed")

        # Show best model so far if we have metrics
        models_with_metrics = [
            r for r in self.results if r.get("status") == "completed" and "metrics" in r
        ]

        if models_with_metrics:
            best_model = max(
                models_with_metrics, key=lambda x: x["metrics"].get("accuracy", 0)
            )
            print(
                f"Best model so far: {best_model['name']} "
                f"(accuracy: {best_model['metrics'].get('accuracy', 0):.3f})"
            )
        print("-" * 40)

    def generate_summary_report(self):
        """Generate a comprehensive comparison report of all models."""
        print("\n[INFO] Generating summary report...")

        # Create summary directory
        summary_dir = Path("outputs") / "model_queue_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        queue_name = self.queue_settings.get("name", "queue")
        report_path = summary_dir / f"{queue_name}_{timestamp}_summary.md"
        csv_path = summary_dir / f"{queue_name}_{timestamp}_metrics.csv"

        # Prepare data for CSV
        rows = []
        for result in self.results:
            row = {
                "name": result["name"],
                "model_type": result["model_type"],
                "status": result["status"],
                "training_time": result.get("training_time", 0),
            }

            # Add metrics if available
            if "metrics" in result:
                metrics = result["metrics"]
                row.update(
                    {
                        "accuracy": metrics.get("accuracy", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1_score": metrics.get("f1_score", 0),
                        "auc": metrics.get("auc", 0),
                    }
                )

            # Add key hyperparameters
            hyperparams = result.get("hyperparameters", {})
            row.update({f"hp_{k}": v for k, v in hyperparams.items()})

            rows.append(row)

        # Save CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Generate Markdown report
        with open(report_path, "w") as f:
            f.write("# Model Queue Summary Report\n\n")
            f.write(f"**Queue Name:** {queue_name}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Models:** {len(self.model_queue)}\n")
            # Calculate actual queue runtime (only if start_time was set during execution)
            total_runtime = 0
            if hasattr(self, "start_time") and self.start_time > 0:
                total_runtime = (time.time() - self.start_time) / 60
                # Only show time if it's reasonable (less than a day)
                if total_runtime < 1440:  # 24 hours
                    f.write(f"**Total Time:** {total_runtime:.2f} minutes\n\n")
                else:
                    f.write("**Total Time:** N/A (report generated separately)\n\n")
            else:
                f.write("**Total Time:** N/A (report generated separately)\n\n")

            # Summary statistics
            completed = [r for r in self.results if r.get("status") == "completed"]
            failed = [r for r in self.results if r.get("status") == "failed"]

            f.write("## Summary Statistics\n\n")
            f.write(f"- **Completed:** {len(completed)}\n")
            f.write(f"- **Failed:** {len(failed)}\n\n")

            # Best performing models
            if completed:
                models_with_metrics = [r for r in completed if "metrics" in r]
                if models_with_metrics:
                    f.write("## Top Performing Models\n\n")

                    # Sort by accuracy
                    sorted_models = sorted(
                        models_with_metrics,
                        key=lambda x: x["metrics"].get("accuracy", 0),
                        reverse=True,
                    )

                    f.write(
                        "| Rank | Model | Type | Accuracy | F1 Score | Training Time |\n"
                    )
                    f.write(
                        "|------|-------|------|----------|----------|---------------|\n"
                    )

                    for i, model in enumerate(sorted_models[:5], 1):
                        metrics = model["metrics"]
                        f.write(
                            f"| {i} | {model['name']} | {model['model_type']} | "
                            f"{metrics.get('accuracy', 0):.3f} | "
                            f"{metrics.get('f1_score', 0):.3f} | "
                            f"{model.get('training_time', 0):.1f}s |\n"
                        )

                    f.write("\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### {result['name']}\n\n")
                f.write(f"- **Model Type:** {result['model_type']}\n")
                f.write(f"- **Status:** {result['status']}\n")

                if result.get("description"):
                    f.write(f"- **Description:** {result['description']}\n")

                if result.get("training_time"):
                    f.write(
                        f"- **Training Time:** {result['training_time']:.1f} seconds\n"
                    )

                # Add hyperparameters
                if "hyperparameters" in result:
                    hyperparams = result["hyperparameters"]
                    if hyperparams:
                        f.write("- **Hyperparameters:**\n")
                        for key, value in hyperparams.items():
                            f.write(f"  - {key}: {value}\n")

                if "metrics" in result and result["metrics"]:
                    metrics = result["metrics"]
                    f.write("- **Performance Metrics:**\n")
                    f.write(f"  - Accuracy: {metrics.get('accuracy', 0):.3f}\n")
                    f.write(f"  - Precision: {metrics.get('precision', 0):.3f}\n")
                    f.write(f"  - Recall: {metrics.get('recall', 0):.3f}\n")
                    f.write(f"  - F1 Score: {metrics.get('f1_score', 0):.3f}\n")

                if result.get("status") == "failed":
                    f.write(f"- **Error:** {result.get('error', 'Unknown error')}\n")

                f.write("\n")

        print(f"[OK] Summary report saved to: {report_path}")
        print(f"[OK] Metrics CSV saved to: {csv_path}")

        return report_path, csv_path


def run_model_queue(queue_config_path: str, resume: bool = True):
    """
    Convenience function to run a model queue.

    Args:
        queue_config_path: Path to queue configuration YAML
        resume: Whether to resume from checkpoint
    """
    manager = ModelQueueManager(queue_config_path)
    manager.run_queue(resume=resume)
