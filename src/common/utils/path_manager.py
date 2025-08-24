"""
Path management utilities for organizing model and data directories.

This module provides comprehensive path management for the TimeFlies project,
implementing the clean 2-level naming convention:
- Level 1: Experiment Type (tissue_model_encoding)
- Level 2: Configuration Details (method_cells_sexes)

Naming Rules:
- Between main parts: _ (underscores)
- Within compound terms: - (hyphens)

Example: data/processed/uncorrected/head_cnn_age/all-genes_all-cells_all-sexes/
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class PathManager:
    """
    A utility class to construct model and data directory paths based on configuration.

    This class centralizes directory path construction logic using the new clean
    2-level naming convention. It automatically generates organized paths for:
    - Processed data storage
    - Model outputs
    - Analysis results
    - Raw data access

    Attributes:
        config: Configuration object containing experiment parameters
        correction_dir: "batch_corrected" or "uncorrected"
        experiment_name: Full experiment path (level1/level2)
        base_experiment: Level 1 name (tissue_model_encoding)
        config_details: Level 2 name (method_cells_sexes)
    """

    def __init__(self, config: Any):
        """
        Initialize PathManager with experiment configuration.

        Args:
            config: Configuration object with experiment parameters including:
                - tissue type, model type, encoding variable
                - batch correction settings
                - gene filtering options
                - cell and sex type filters
                - train-test split configuration

        Example:
            config = load_config("configs/head_cnn_age.yaml")
            path_manager = PathManager(config)
            model_dir = path_manager.construct_model_directory()
        """
        self.config = config

        # Extract core parameters with proper error handling
        try:
            # Batch correction setting
            self.batch_correction_enabled = getattr(
                config.data.batch_correction, "enabled", False
            )
            self.correction_dir = (
                "batch_corrected" if self.batch_correction_enabled else "uncorrected"
            )

            # Core experiment parameters
            self.tissue = getattr(config.data, "tissue", "head").lower()
            self.model_type = getattr(config.data, "model", "CNN").lower()
            self.target_variable = getattr(
                config.data, "target_variable", "age"
            ).lower()
            self.cell_type = getattr(config.data, "cell_type", "all").lower()
            self.sex_type = getattr(config.data, "sex_type", "all").lower()

            # Generate experiment naming components
            self.experiment_name = self._generate_experiment_name()

        except AttributeError as e:
            raise ValueError(f"Invalid configuration structure: {e}")

    def _get_project_root(self) -> Path:
        """
        Find project root directory by looking for key directories.

        Returns:
            Path: Project root directory

        Raises:
            FileNotFoundError: If project root cannot be located
        """
        # First check if we're in a test environment with a temporary directory
        current_working_dir = Path(os.getcwd())

        # Check if we're running tests (pytest sets PYTEST_CURRENT_TEST)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            # In test mode, use tests directory as project root for outputs
            tests_dir = Path(__file__).parent.parent.parent.parent / "tests"
            if tests_dir.exists():
                return tests_dir
            # Fallback to current working directory if tests dir not found
            return current_working_dir

        # For deployment: If current working directory has data, use it as project root
        # This allows users to run TimeFlies from their own project directories
        if (current_working_dir / "data").exists():
            return current_working_dir

        # For deployment: Use current working directory if we're not in development
        # In development, we have the src/common/utils structure
        is_development = (
            Path(__file__).parent.parent.name == "common"
            and Path(__file__).parent.parent.parent.name == "src"
        )

        if not is_development:
            return current_working_dir

        # Otherwise, search from the file location for the actual project root
        current_dir = Path(__file__).parent.absolute()

        # Search up to 5 levels for project root indicators
        for _ in range(5):
            # Look for key directories that indicate project root
            # Require both 'src' and at least one of the other indicators to ensure we find the actual project root
            if (current_dir / "src").exists() and any(
                (current_dir / indicator).exists()
                for indicator in ["data", "outputs", "configs"]
            ):
                return current_dir
            current_dir = current_dir.parent

        raise FileNotFoundError(
            "Could not locate TimeFlies project root. "
            "Make sure you're running from within the project directory."
        )

    def _generate_experiment_name(self) -> str:
        """
        Generate clean experiment name using 2-level naming convention.

        Returns:
            str: Full experiment path (level1/level2)

        Example:
            head_cnn_age/all-genes_all-cells_all-sexes
        """
        # Level 1: Basic experiment (tissue_model_encoding)
        self.base_experiment = "_".join(
            [self.tissue, self.model_type, self.target_variable]
        )

        # Level 2: Configuration details (method_cells_sexes)
        config_parts = [self._get_gene_method()]

        # Add cell type with proper formatting
        if self.cell_type != "all":
            # Convert spaces to hyphens within cell type names
            cell_type_clean = self.cell_type.replace(" ", "-")
            config_parts.append(cell_type_clean)
        else:
            config_parts.append("all-cells")

        # Add sex type
        if self.sex_type != "all":
            config_parts.append(self.sex_type)
        else:
            config_parts.append("all-sexes")

        # Join configuration parts with underscores
        self.config_details = "_".join(config_parts)

        # Add split configuration to model path for unique models per split
        try:
            from .split_naming import SplitNamingUtils

            split_name = SplitNamingUtils.generate_split_name(
                {
                    "method": getattr(self.config.data.split, "method", "random"),
                    "test_ratio": getattr(self.config.data.split, "test_ratio", 0.2),
                    "train": getattr(self.config.data.split, "train", []),
                    "test": getattr(self.config.data.split, "test", []),
                }
            )

            # Add split suffix to config details for unique model paths per split
            self.config_details = f"{self.config_details}_{split_name}"

        except (AttributeError, ImportError):
            # Fallback if split configuration is not available
            pass

        # Return combined path
        return f"{self.base_experiment}/{self.config_details}"

    def _get_gene_method(self) -> str:
        """Get gene preprocessing method name."""
        try:
            gene_filtering = self.config.preprocessing.genes
            gene_balancing = self.config.preprocessing.balancing

            # Check for specific methods
            if getattr(gene_filtering, "highly_variable_genes", False):
                return "hvg"
            elif getattr(gene_balancing, "balance_genes", False):
                return "balanced"
            elif getattr(gene_balancing, "balance_lnc_genes", False):
                return "balanced-lnc"
            elif getattr(gene_filtering, "select_batch_genes", False):
                return "batch-genes"
            elif getattr(gene_filtering, "only_keep_lnc_genes", False):
                return "only-lnc"
            elif getattr(gene_filtering, "remove_lnc_genes", False):
                return "no-lnc"
            elif getattr(gene_filtering, "remove_autosomal_genes", False):
                return "no-autosomal"
            elif getattr(gene_filtering, "remove_sex_genes", False):
                return "no-sex"
            else:
                return "all-genes"
        except AttributeError:
            # Fallback if config structure is different
            return "all-genes"

    def construct_model_directory(self) -> str:
        """
        Get the experiment directory path for model storage.

        Returns:
            str: The experiment directory path
        """
        return self.get_experiment_dir()

    def get_visualization_directory(self, subfolder: str | None = None) -> str:
        """
        Get the evaluation directory path for visualizations.

        Args:
            subfolder: Additional subfolder (e.g., 'plots')

        Returns:
            str: The experiment evaluation directory path
        """
        experiment_dir = self.get_experiment_dir()
        eval_dir = os.path.join(experiment_dir, "evaluation")

        if subfolder:
            eval_dir = os.path.join(eval_dir, subfolder)

        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir

    def get_processed_data_dir(self) -> str:
        """
        Constructs the directory path for saving or loading preprocessed data.

        Returns:
            str: The path to the processed data directory

        Example:
            data/processed/batch_corrected/head_cnn_age/all-genes_all-cells_all-sexes/
        """
        project_root = self._get_project_root()

        # Use clean 2-level structure: data/processed/correction_dir/level1/level2/
        processed_data_dir = (
            project_root
            / "data"
            / "processed"
            / self.correction_dir
            / self.base_experiment
            / self.config_details
        )

        # Create directory if it doesn't exist
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        return str(processed_data_dir)

    def get_raw_data_dir(self, tissue_override: str | None = None) -> str:
        """
        Constructs the directory path for raw h5ad data files using new project structure.

        Args:
            tissue_override: Override the tissue from config (e.g., for cross-tissue analysis)

        Returns:
            str: The path to the raw data directory

        Example:
            data/fruitfly_aging/head/
        """
        project_root = self._get_project_root()
        tissue = tissue_override or self.tissue

        # New project-specific data structure: data/project/tissue/
        project = getattr(self.config, "project", "fruitfly_aging")
        raw_data_dir = project_root / "data" / project / tissue

        return str(raw_data_dir)

    def get_log_directory(self) -> str:
        """
        Constructs the directory path for log files.

        Returns:
            str: The path to the logs directory
        """
        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")
        log_dir = project_root / "outputs" / project_name / "logs"

        # Create directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        return str(log_dir)

    def get_outputs_directory(self) -> Path:
        """
        Get the project-specific outputs directory root.

        Returns:
            Path: The path to the project outputs directory

        Example:
            outputs/fruitfly_aging/
        """
        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")
        outputs_dir = project_root / "outputs" / project_name

        # Create directory if it doesn't exist (skip during tests)
        import os

        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            outputs_dir.mkdir(parents=True, exist_ok=True)

        return outputs_dir

    def get_training_visuals_dir(self) -> str:
        """
        Get the training visuals directory within the model directory.

        Returns:
            str: Path to training visuals directory

        Example:
            outputs/fruitfly_alzheimers/models/uncorrected/head_cnn_age/all-genes/training/visuals/
        """
        model_dir = self.construct_model_directory()
        training_visuals_dir = Path(model_dir) / "training" / "visuals"
        training_visuals_dir.mkdir(parents=True, exist_ok=True)
        return str(training_visuals_dir)

    # =============================================================================
    # NEW EXPERIMENT-BASED STRUCTURE METHODS
    # =============================================================================

    def generate_experiment_name(self) -> str:
        """
        Generate timestamp for experiment within config directory.

        Returns:
            str: Timestamp like "2025-01-22_14-30-15"
        """
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def get_config_key(self) -> str:
        """
        Get the config-specific directory name including all relevant settings.

        Returns:
            str: Config key for grouping experiments

        Examples:
            "head_cnn_age_ctrl-vs-alz"
            "body_mlp_age_all"
            "head_xgboost_age_m-vs-f"
        """
        from .split_naming import SplitNamingUtils

        # Build key parts
        parts = []

        # Tissue (if not default head)
        if self.tissue != "head":
            parts.append(self.tissue)
        else:
            parts.append("head")  # Always include tissue for clarity

        # Model type
        parts.append(self.model_type)

        # Target variable (if not default age)
        if self.target_variable != "age":
            parts.append(self.target_variable)
        else:
            parts.append("age")  # Always include for clarity

        # Split configuration
        experiment_suffix = SplitNamingUtils.generate_experiment_suffix(self.config)
        parts.append(experiment_suffix)

        # Optional: Add gene filtering if special
        if getattr(self.config.preprocessing.genes, "highly_variable_genes", False):
            parts.append("hvg")
        elif getattr(self.config.preprocessing.balancing, "balance_genes", False):
            parts.append("balanced")

        return "_".join(parts)

    def get_experiment_dir(self, experiment_name: str = None) -> str:
        """
        Get experiment directory path within config-specific folder.

        Args:
            experiment_name: Specific experiment timestamp, or None for current

        Returns:
            str: Path to experiment directory

        Example:
            outputs/fruitfly_alzheimers/experiments/uncorrected/cnn_ctrl-vs-alz/2025-01-22_14-30-15/
        """
        if experiment_name is None:
            experiment_name = self.generate_experiment_name()

        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")

        # Add batch correction directory level
        batch_correction_enabled = getattr(
            self.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        # Group by config key (e.g., cnn_ctrl-vs-alz)
        config_key = self.get_config_key()

        # Store all real experiments in all_runs directory
        experiment_dir = (
            project_root
            / "outputs"
            / project_name
            / "experiments"
            / correction_dir
            / "all_runs"
            / config_key
            / experiment_name
        )
        return str(experiment_dir)

    def get_experiment_model_path(self, experiment_name: str = None) -> str:
        """Get path to model.h5 in experiment directory."""
        experiment_dir = self.get_experiment_dir(experiment_name)
        return str(Path(experiment_dir) / "model.h5")

    def get_experiment_training_dir(self, experiment_name: str = None) -> str:
        """Get training directory in experiment."""
        experiment_dir = self.get_experiment_dir(experiment_name)
        return str(Path(experiment_dir) / "training")

    def get_experiment_evaluation_dir(self, experiment_name: str = None) -> str:
        """Get evaluation directory in experiment."""
        experiment_dir = self.get_experiment_dir(experiment_name)
        return str(Path(experiment_dir) / "evaluation")

    def get_experiment_plots_dir(self, experiment_name: str = None) -> str:
        """Get plots directory in experiment."""
        evaluation_dir = self.get_experiment_evaluation_dir(experiment_name)
        return str(Path(evaluation_dir) / "plots")

    def create_experiment_metadata(self, experiment_name: str = None) -> dict:
        """
        Create metadata dictionary for experiment.

        Returns:
            dict: Experiment metadata
        """
        from .split_naming import SplitNamingUtils

        if experiment_name is None:
            experiment_name = self.generate_experiment_name()

        return {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type.upper(),
            "target": self.target_variable,
            "tissue": self.tissue,
            "data_filters": {
                "genes": "highly_variable"
                if getattr(
                    self.config.preprocessing.genes, "highly_variable_genes", False
                )
                else "all",
                "cells": self.cell_type,
                "sex": self.sex_type,
                "samples": getattr(self.config.data.sampling, "samples", None),
                "variables": getattr(self.config.data.sampling, "variables", None),
            },
            "batch_correction": self.batch_correction_enabled,
            "split_config": SplitNamingUtils.extract_split_details_for_metadata(
                self.config
            ),
        }

    def save_experiment_metadata(
        self, experiment_name: str = None, additional_data: dict = None
    ):
        """
        Save metadata.json for experiment.

        Args:
            experiment_name: Experiment name
            additional_data: Additional metadata to include
        """
        if experiment_name is None:
            experiment_name = self.generate_experiment_name()

        metadata = self.create_experiment_metadata(experiment_name)
        if additional_data:
            metadata.update(additional_data)

        experiment_dir = Path(self.get_experiment_dir(experiment_name))
        experiment_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_best_symlink_path(self) -> str:
        """Get best symlink path within config directory."""
        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")

        # Add batch correction directory level
        batch_correction_enabled = getattr(
            self.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        # Best symlink is inside the config directory
        config_key = self.get_config_key()

        return str(
            project_root
            / "outputs"
            / project_name
            / "experiments"
            / correction_dir
            / config_key
            / "best"
        )

    def get_latest_symlink_path(self) -> str:
        """Get latest symlink path within correction directory."""
        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")

        # Add batch correction directory level
        batch_correction_enabled = getattr(
            self.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        return str(
            project_root
            / "outputs"
            / project_name
            / "experiments"
            / correction_dir
            / "latest"
        )

    def update_latest_symlink(self, experiment_name: str):
        """
        Update 'latest' symlink to point to current experiment.

        Args:
            experiment_name: Name of experiment to link to
        """
        latest_path = Path(self.get_latest_symlink_path())
        self.get_experiment_dir(experiment_name)

        # Remove existing symlink
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()

        # Create new symlink pointing to all_runs/config_key/experiment_name
        config_key = self.get_config_key()
        relative_path = Path("all_runs") / config_key / experiment_name
        latest_path.symlink_to(relative_path)

    def update_best_symlink(self, experiment_name: str):
        """
        Update best model symlink in the best/ collection directory at correction level.

        Args:
            experiment_name: Timestamp of experiment (e.g., "2025-01-22_14-30-15")
        """
        # Get paths
        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")
        correction_dir = (
            "batch_corrected" if self.batch_correction_enabled else "uncorrected"
        )
        config_key = self.get_config_key()

        # Create best/ directory at correction level
        best_dir = (
            project_root
            / "outputs"
            / project_name
            / "experiments"
            / correction_dir
            / "best"
        )
        best_dir.mkdir(parents=True, exist_ok=True)

        # Path to symlink for this config within best/
        best_config_link = best_dir / config_key

        # Remove existing symlink for this config
        if best_config_link.exists() or best_config_link.is_symlink():
            best_config_link.unlink()

        # Create new symlink: best/{config_key} -> ../all_runs/{config_key}/{experiment_name}
        relative_path = Path("../all_runs") / config_key / experiment_name
        best_config_link.symlink_to(relative_path)

        print(
            f"✓ Updated best model collection: best/{config_key} → {config_key}/{experiment_name}"
        )

    def get_best_model_dir_for_config(self) -> str:
        """
        Get the best model directory for the current configuration from best/ collection.

        Returns:
            str: Path to best model experiment directory for current config
        """
        # Get paths
        project_root = self._get_project_root()
        project_name = getattr(self.config, "project", "fruitfly_aging")
        correction_dir = (
            "batch_corrected" if self.batch_correction_enabled else "uncorrected"
        )
        config_key = self.get_config_key()

        # Look for symlink in best/ directory
        best_config_link = (
            project_root
            / "outputs"
            / project_name
            / "experiments"
            / correction_dir
            / "best"
            / config_key
        )

        if best_config_link.exists() and best_config_link.is_symlink():
            # Follow symlink to get experiment directory
            # The symlink points to: ../all_runs/{config_key}/{experiment_name}
            experiment_dir = best_config_link.parent / best_config_link.readlink()
            return str(experiment_dir.resolve())
        else:
            # No best model exists yet for this config
            return None
