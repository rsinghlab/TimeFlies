# Inital setup for stratified splitting of AnnData objects and verification of the splits. Needs to be run before training the models.

import numpy as np
from pathlib import Path
import anndata
import logging
from typing import List
import pandas as pd
from ..core.config_manager import ConfigManager

# Configure logging for better control over output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSetupManager:
    """
    A class to perform initial setup tasks such as stratified splitting of AnnData objects
    and verifying the integrity of the splits.
    """

    def __init__(self, config=None):
        """
        Initialize the DataSetupManager class with configuration.

        Args:
            config: Configuration instance from ConfigManager. If None, creates default config.
        """
        if config is None:
            try:
                self.config = ConfigManager()
            except Exception:
                # If no config file available, create minimal config
                minimal_config = {
                    "general": {
                        "project_name": "TimeFlies",
                        "version": "0.2.0",
                        "random_state": 42,
                    },
                    "data": {
                        "tissue": "head",
                        "batch_correction": {"enabled": False},
                        "sampling": {"num_samples": None, "num_variables": None},
                        "train_test_split": {"test_split": 0.2, "random_state": 42},
                        "encoding_variable": "age",
                    },
                }
                from ..core.config_manager import Config

                self.config = Config(minimal_config)
        else:
            self.config = config

    def _generate_filename(
        self, tissue: str, file_type: str, batch: bool = False
    ) -> str:
        """
        Generate filename based on project configuration using new naming convention.

        Format: {species}_{tissue}_{project}_{type}[_batch].h5ad
        """
        species = getattr(self.config.data, "species", "drosophila")
        project = getattr(self.config.data, "project", "fruitfly_aging").replace(
            "fruitfly_", ""
        )

        filename = f"{species}_{tissue}_{project}_{file_type}"
        if batch:
            filename += "_batch"
        filename += ".h5ad"
        return filename

    def setup_data(self):
        """Perform complete data setup including downloading, splitting, and verification."""

        logger.info("Starting data setup process...")

        # Step 1: Download data (if needed)
        try:
            self.download_data()
        except Exception as e:
            logger.error(f"Data download failed: {e}")
            return False

        # Step 2: Perform stratified split
        try:
            self.stratified_split()
        except Exception as e:
            logger.error(f"Stratified split failed: {e}")
            return False

        # Step 3: Verify splits
        try:
            self.verify_splits()
        except Exception as e:
            logger.error(f"Split verification failed: {e}")
            return False

        logger.info("Data setup completed successfully!")
        return True

    def download_data(self):
        """Download and prepare data files."""

        tissue = self.config.data.tissue
        data_dir = Path(
            f"data/{getattr(self.config.data, 'project', 'fruitfly_aging')}/{tissue}"
        )
        data_dir.mkdir(parents=True, exist_ok=True)

        original_file = data_dir / self._generate_filename(tissue, "original")

        if original_file.exists():
            logger.info(f"Data file already exists: {original_file}")
            return True

        # Here you would implement actual download logic
        # For now, just log that this step needs implementation
        logger.warning(
            f"Data download not implemented. Please ensure {original_file} exists."
        )
        return original_file.exists()

    def stratified_split(self):
        """Perform stratified splitting of the dataset."""

        logger.info("Performing stratified split...")

        tissue = self.config.data.tissue
        data_dir = Path(
            f"data/{getattr(self.config.data, 'project', 'fruitfly_aging')}/{tissue}"
        )

        # File paths using new naming convention
        original_file = data_dir / self._generate_filename(tissue, "original")
        train_file = data_dir / self._generate_filename(tissue, "train")
        eval_file = data_dir / self._generate_filename(tissue, "eval")

        if not original_file.exists():
            raise FileNotFoundError(f"Original data file not found: {original_file}")

        if train_file.exists() and eval_file.exists():
            logger.info("Split files already exist. Skipping stratified split.")
            return True

        # Load the original data
        logger.info(f"Loading data from: {original_file}")
        adata = anndata.read_h5ad(original_file)

        # Get stratification parameters
        encoding_var = self.config.data.encoding_variable
        test_split = self.config.data.train_test_split.test_split
        random_state = self.config.data.train_test_split.random_state

        # Perform stratified split
        from sklearn.model_selection import train_test_split

        # Get stratification labels
        if encoding_var not in adata.obs.columns:
            raise ValueError(f"Encoding variable '{encoding_var}' not found in data.")

        labels = adata.obs[encoding_var].values
        indices = np.arange(len(labels))

        # Perform stratified split
        train_idx, eval_idx = train_test_split(
            indices, test_size=test_split, stratify=labels, random_state=random_state
        )

        # Create and save splits
        adata_train = adata[train_idx].copy()
        adata_eval = adata[eval_idx].copy()

        logger.info(f"Saving training data to: {train_file}")
        adata_train.write_h5ad(train_file)

        logger.info(f"Saving evaluation data to: {eval_file}")
        adata_eval.write_h5ad(eval_file)

        logger.info(
            f"Split completed: {len(adata_train)} train, {len(adata_eval)} eval samples"
        )
        return True

    def verify_splits(self):
        """Verify the integrity of the data splits."""

        logger.info("Verifying data splits...")

        tissue = self.config.data.tissue
        data_dir = Path(
            f"data/{getattr(self.config.data, 'project', 'fruitfly_aging')}/{tissue}"
        )

        # File paths
        original_file = data_dir / self._generate_filename(tissue, "original")
        train_file = data_dir / self._generate_filename(tissue, "train")
        eval_file = data_dir / self._generate_filename(tissue, "eval")

        # Check if all files exist
        for file_path in [original_file, train_file, eval_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load data
        adata_original = anndata.read_h5ad(original_file)
        adata_train = anndata.read_h5ad(train_file)
        adata_eval = anndata.read_h5ad(eval_file)

        # Verify sample counts
        total_samples = len(adata_original)
        train_samples = len(adata_train)
        eval_samples = len(adata_eval)

        if train_samples + eval_samples != total_samples:
            raise ValueError(
                f"Sample count mismatch: {train_samples} + {eval_samples} != {total_samples}"
            )

        # Verify no overlap in cell indices
        train_obs_names = set(adata_train.obs_names)
        eval_obs_names = set(adata_eval.obs_names)

        if train_obs_names & eval_obs_names:  # Intersection should be empty
            raise ValueError("Found overlapping samples between train and eval sets")

        # Verify stratification worked
        encoding_var = self.config.data.encoding_variable
        original_dist = (
            adata_original.obs[encoding_var].value_counts(normalize=True).sort_index()
        )
        train_dist = (
            adata_train.obs[encoding_var].value_counts(normalize=True).sort_index()
        )
        eval_dist = (
            adata_eval.obs[encoding_var].value_counts(normalize=True).sort_index()
        )

        logger.info("Distribution verification:")
        logger.info(f"Original: {dict(original_dist)}")
        logger.info(f"Train: {dict(train_dist)}")
        logger.info(f"Eval: {dict(eval_dist)}")

        # Check if distributions are similar (within reasonable tolerance)
        tolerance = 0.05  # 5% tolerance
        for label in original_dist.index:
            orig_prop = original_dist[label]
            train_prop = train_dist.get(label, 0)
            eval_prop = eval_dist.get(label, 0)

            if (
                abs(train_prop - orig_prop) > tolerance
                or abs(eval_prop - orig_prop) > tolerance
            ):
                logger.warning(
                    f"Distribution difference for {label}: orig={orig_prop:.3f}, train={train_prop:.3f}, eval={eval_prop:.3f}"
                )

        logger.info("Split verification completed successfully!")
        return True

    def get_data_info(self):
        """Get information about the current data setup."""

        tissue = self.config.data.tissue
        data_dir = Path(
            f"data/{getattr(self.config.data, 'project', 'fruitfly_aging')}/{tissue}"
        )

        files_info = {}

        for file_type in ["original", "train", "eval"]:
            file_path = data_dir / self._generate_filename(tissue, file_type)

            if file_path.exists():
                try:
                    adata = anndata.read_h5ad(file_path)
                    files_info[file_type] = {
                        "path": str(file_path),
                        "samples": len(adata),
                        "genes": len(adata.var),
                        "exists": True,
                    }
                except Exception as e:
                    files_info[file_type] = {
                        "path": str(file_path),
                        "error": str(e),
                        "exists": True,
                    }
            else:
                files_info[file_type] = {"path": str(file_path), "exists": False}

        return files_info
