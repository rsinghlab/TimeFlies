# Inital setup for stratified splitting of AnnData objects and verification of the splits. Needs to be run before training the models.

import numpy as np
from pathlib import Path
import anndata
import logging
from typing import List
import pandas as pd
from ...core.config_manager import ConfigManager

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
            self.config = ConfigManager.create_default()
        else:
            self.config = config
        logger.info("DataSetupManager initialized with configuration.")

    def _resolve_data_path(self, tissue: str, batch: bool) -> Path:
        """
        Resolve the path to the data directory based on tissue type and batch correction.

        Args:
            tissue (str): The tissue type (e.g., 'head', 'body').
            batch (bool): Whether to use batch-corrected data.

        Returns:
            Path: The resolved data directory path.
        """
        # Get the directory of the current script
        code_dir = Path(__file__).resolve().parent

        # Construct the base data directory path
        data_dir = code_dir.parent / "TimeFlies" / "Data" / "h5ad" / tissue

        # Append 'batch_corrected' or 'uncorrected' based on the batch flag
        data_dir /= "batch_corrected" if batch else "uncorrected"

        logger.debug(f"Resolved data directory: {data_dir}")
        return data_dir

    def _load_anndata(self, file_path: Path) -> anndata.AnnData:
        """
        Load an AnnData object from a given file path.

        Args:
            file_path (Path): Path to the .h5ad file.

        Returns:
            anndata.AnnData: The loaded AnnData object.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the AnnData object and return a copy to avoid unintended modifications
        adata = anndata.read_h5ad(file_path).copy()
        logger.info(f"Loaded data from {file_path}")
        return adata

    def _select_samples(
        self, strata: pd.Series, split_size: int, random_seed: int
    ) -> List[str]:
        """
        Select samples for the evaluation set based on stratification.

        Args:
            strata (pd.Series): The stratification column.
            split_size (int): Number of samples to split for evaluation.
            random_seed (int): Seed for reproducibility.

        Returns:
            List[str]: List of selected sample indices.
        """
        # Set the random seed for reproducibility
        np.random.seed(random_seed)

        # Calculate the proportion of each class
        proportions = strata.value_counts(normalize=True)
        logger.debug(f"Stratification proportions:\n{proportions}")

        # Determine the number of samples to take from each class
        n_samples = (proportions * split_size).round().astype(int)
        logger.debug(f"Number of samples per class:\n{n_samples}")

        selected_samples = []

        # Iterate over each class and randomly select the required number of samples
        for category, n in n_samples.items():
            samples = strata.index[strata == category]
            if len(samples) < n:
                logger.warning(
                    f"Not enough samples for category '{category}'. "
                    f"Requested: {n}, available: {len(samples)}. Selecting all available samples."
                )
                n = len(samples)  # Adjust to available samples

            # Randomly choose samples without replacement
            selected = np.random.choice(samples, size=n, replace=False)
            selected_samples.extend(selected)

            logger.debug(f"Selected {n} samples for category '{category}'.")

        logger.info(f"Total selected samples for evaluation: {len(selected_samples)}")
        return selected_samples

    def stratified_split_and_save(
        self,
        strata_column: str,
        tissue: str,
        random_seed: int,
        batch: bool,
        split_size: int,
    ) -> None:
        """
        Perform a stratified split on an AnnData object and save the resulting training and evaluation datasets.

        Args:
            strata_column (str): The column to use for stratification.
            tissue (str): Tissue type to define the directory for saving.
            random_seed (int): Seed for random number generator.
            batch (bool): Whether to use batch-corrected data.
            split_size (int): Number of samples to split for evaluation.

        Returns:
            None
        """
        logger.info(f"Starting stratified split for tissue '{tissue}', batch={batch}.")

        # Resolve the data directory path
        data_dir = self._resolve_data_path(tissue, batch)

        # Define the path to the original .h5ad file
        h5ad_file_path = data_dir / "fly_original.h5ad"

        # Load the AnnData object
        adata = self._load_anndata(h5ad_file_path)

        # Define the stratification variable
        strata = adata.obs[strata_column]
        logger.debug(
            f"Stratification column '{strata_column}' loaded with {strata.nunique()} unique classes."
        )

        # Select samples for evaluation set based on stratification
        selected_samples = self._select_samples(strata, split_size, random_seed)

        # Create new AnnData objects for training and evaluation sets
        adata_eval = adata[selected_samples, :].copy()
        adata_train = adata[~adata.obs_names.isin(selected_samples), :].copy()

        logger.info(f"Created evaluation dataset with {adata_eval.n_obs} samples.")
        logger.info(f"Created training dataset with {adata_train.n_obs} samples.")

        # Define paths for the split datasets
        fly_train_path = data_dir / "fly_train.h5ad"
        fly_eval_path = data_dir / "fly_eval.h5ad"

        # Save the split datasets
        adata_train.write_h5ad(fly_train_path)
        adata_eval.write_h5ad(fly_eval_path)

        logger.info(f"Saved training data to {fly_train_path}")
        logger.info(f"Saved evaluation data to {fly_eval_path}")

    def verify(
        self,
        tissue: str,
        batch: bool,
        strata_column: str,
    ) -> bool:
        """
        Verify that the training and evaluation datasets independently maintain the same stratification proportions as the original dataset.

        Args:
            tissue (str): Tissue type.
            batch (bool): Whether to use batch-corrected data.
            strata_column (str): Column used for stratification.

        Returns:
            bool: True if verification passes for both datasets, False otherwise.
        """
        logger.info(f"Starting verification for tissue '{tissue}', batch={batch}.")

        # Resolve the data directory path
        data_dir = self._resolve_data_path(tissue, batch)

        # Define paths to the split datasets and original dataset
        fly_eval_path = data_dir / "fly_eval.h5ad"
        fly_train_path = data_dir / "fly_train.h5ad"
        original_path = data_dir / "fly_original.h5ad"

        # Load the AnnData objects
        try:
            eval_adata = self._load_anndata(fly_eval_path)
            train_adata = self._load_anndata(fly_train_path)
            original_adata = self._load_anndata(original_path)
        except FileNotFoundError:
            logger.error("One or more required files are missing for verification.")
            return False

        # Verify no overlap between evaluation and training datasets
        eval_samples = set(eval_adata.obs_names)
        train_samples = set(train_adata.obs_names)
        overlap = eval_samples.intersection(train_samples)

        if overlap:
            logger.error(
                f"Overlap detected between eval and train datasets for tissue '{tissue}', batch={batch}. "
                f"Overlapping samples: {overlap}"
            )
            return False
        else:
            logger.info(
                f"No overlap between eval and train datasets for tissue '{tissue}', batch={batch}."
            )

        # Since strata_columns is now a single column, simplify the aggregation
        # Get counts for each stratum in the original, training, and evaluation datasets
        original_counts = original_adata.obs[strata_column].value_counts().sort_index()
        train_counts = train_adata.obs[strata_column].value_counts().sort_index()
        eval_counts = eval_adata.obs[strata_column].value_counts().sort_index()

        logger.debug(f"Original strata counts:\n{original_counts}")
        logger.debug(f"Training strata counts:\n{train_counts}")
        logger.debug(f"Evaluation strata counts:\n{eval_counts}")

        # Calculate proportions
        total_original = original_counts.sum()
        total_train = train_counts.sum()
        total_eval = eval_counts.sum()

        original_proportions = original_counts / total_original
        train_proportions = (
            train_counts / total_train if total_train > 0 else pd.Series()
        )
        eval_proportions = eval_counts / total_eval if total_eval > 0 else pd.Series()

        logger.debug(f"Original strata proportions:\n{original_proportions}")
        logger.debug(f"Training strata proportions:\n{train_proportions}")
        logger.debug(f"Evaluation strata proportions:\n{eval_proportions}")

        # Define a tolerance level for the stratification check (e.g., 5%)
        tolerance = 0.05

        # Function to compare proportions
        def check_proportions(
            original: pd.Series, split: pd.Series, split_name: str
        ) -> bool:
            for strata, orig_prop in original.items():
                split_prop = split.get(strata, 0)
                difference = abs(orig_prop - split_prop)
                if difference > tolerance:
                    logger.error(
                        f"Stratification proportion for '{strata}' in {split_name} dataset differs by {difference:.2f}, which exceeds the tolerance of {tolerance}."
                    )
                    return False
            return True

        # Check proportions in training dataset
        train_check = check_proportions(
            original_proportions, train_proportions, "training"
        )
        # Check proportions in evaluation dataset
        eval_check = check_proportions(
            original_proportions, eval_proportions, "evaluation"
        )

        if train_check and eval_check:
            logger.info(
                f"Stratification integrity maintained for both training and evaluation datasets for tissue '{tissue}', batch={batch}."
            )
            return True
        else:
            logger.error(
                f"Stratification integrity check failed for tissue '{tissue}', batch={batch}."
            )
            return False

    def main(self):
        """
        Execute the stratified split and verification based on the external configuration.

        The method performs the following steps:
            1. Extracts parameters from the configuration.
            2. Performs the stratified split and saves the datasets.
            3. Verifies the integrity of the split datasets.
        """
        logger.info("Starting main execution of InitialSetup.")

        # Extract split parameters from the configuration
        strata_column = getattr(self.config.data, 'strata_column', 'age')
        tissue = getattr(self.config.data, 'tissue', 'head') 
        seed = getattr(self.config.general, 'random_state', 42)
        batch = getattr(self.config.data.batch_correction, 'enabled', False)
        split_size = getattr(self.config.data.train_test_split, 'split_size', 1000)

        logger.info(
            f"Configuration - Strata: {strata_column}, Tissue: {tissue}, Seed: {seed}, Batch: {batch}, Split Size: {split_size}"
        )

        # Perform the stratified split and save the datasets
        try:
            self.stratified_split_and_save(
                strata_column=strata_column,
                tissue=tissue,
                random_seed=seed,
                batch=batch,
                split_size=split_size,
            )
            logger.info("Stratified split and save completed successfully.")
        except Exception as e:
            logger.exception(f"Error during stratified split and save: {e}")
            return

        # Perform verification
        try:
            verification_result = self.verify(
                tissue=tissue, batch=batch, strata_column=strata_column
            )
            if verification_result:
                logger.info("Verification successful.")
            else:
                logger.error("Verification failed.")
        except Exception as e:
            logger.exception(f"Error during verification: {e}")

        logger.info("Main execution of DataSetupManager completed.")
    
    def run(self):
        """Simple wrapper around main for CLI integration.""" 
        self.main()


if __name__ == "__main__":
    setup = DataSetupManager()
    setup.main()
