# Inital setup for stratified splitting of AnnData objects and verification of the splits. Needs to be run before training the models.

import logging
from pathlib import Path

import anndata
import numpy as np

from common.core import ConfigManager

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
                config_manager = ConfigManager()
                self.config = config_manager.get_config()
            except Exception:
                # If no config file available, create minimal config
                minimal_config = {
                    "general": {
                        "project_name": "TimeFlies",
                        "version": "1.0.0",
                        "random_state": 42,
                    },
                    "data": {
                        "tissue": "head",
                        "batch_correction": {"enabled": False},
                        "sampling": {"num_samples": None, "num_variables": None},
                        "train_test_split": {"test_split": 0.2, "random_state": 42},
                        "target_variable": "age",
                    },
                }
                from common.core import Config

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
        """Perform complete data setup including downloading, splitting, and verification for all projects."""

        logger.info("Starting data setup process...")

        # Auto-detect all projects in data/ folder
        data_dir = Path("data")
        if not data_dir.exists():
            logger.error(
                "Data directory not found. Please create data/ and add your H5AD files."
            )
            return False

        projects = [p.name for p in data_dir.iterdir() if p.is_dir()]
        if not projects:
            logger.error(
                "No project directories found in data/. Please add project folders like data/fruitfly_aging/"
            )
            return False

        logger.info(f"Found projects: {projects}")

        # Process each project
        success_count = 0
        for project in projects:
            logger.info(f"Processing project: {project}")
            try:
                # Step 1: Download data (if needed)
                if not self.download_data_for_project(project):
                    logger.warning(f"Data download failed for {project}")
                    continue

                # Step 2: Perform stratified split
                if not self.stratified_split():
                    logger.warning(f"Stratified split failed for {project}")
                    continue

                # Step 3: Verify splits
                if not self.verify_splits():
                    logger.warning(f"Split verification failed for {project}")
                    continue

                logger.info(f"✅ {project} setup completed successfully!")
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {project}: {e}")
                continue

        if success_count == 0:
            logger.error("No projects were processed successfully!")
            return False
        elif success_count < len(projects):
            logger.warning(
                f"Only {success_count}/{len(projects)} projects processed successfully"
            )
        else:
            logger.info("All projects setup completed successfully!")

        return success_count > 0

    def download_data_for_project(self, project_name):
        """Download and prepare data files for a specific project."""
        tissue = self.config.data.tissue
        data_dir = Path(f"data/{project_name}/{tissue}")

        if not data_dir.exists():
            logger.info(f"No {tissue} data directory found for {project_name}")
            return True  # Skip if no data directory

        # Look for any original files in the directory
        original_files = list(data_dir.glob("*original*.h5ad"))

        if original_files:
            logger.info(
                f"Data files already exist for {project_name}: {[f.name for f in original_files]}"
            )
            return True

        # Here you would implement actual download logic
        logger.warning(f"No original data files found in {data_dir}")
        return len(original_files) > 0

    def download_data(self):
        """Backward compatibility - download for current config project."""
        project = getattr(self.config.data, "project", "fruitfly_aging")
        return self.download_data_for_project(project)

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

        # Load setup configuration
        setup_config_path = Path("configs/setup.yaml")
        if setup_config_path.exists():
            import yaml

            with open(setup_config_path) as f:
                setup_config = yaml.safe_load(f)

            # Use setup.yaml for split configuration
            split_config = setup_config.get("train_test_split", {})
            split_size = split_config.get("split_size", 5000)
            stratify_by = split_config.get("stratify_by", ["age"])
            # Handle both string and list format
            if isinstance(stratify_by, str):
                stratify_by = [stratify_by]
            fallback_ratio = split_config.get("fallback_ratio", 0.2)
            random_state = split_config.get("random_state", 42)
            logger.info(
                f"Using setup.yaml: split_size={split_size}, stratify_by={stratify_by}"
            )
        else:
            # Fallback to defaults
            split_size = 5000
            stratify_by = ["age"]
            fallback_ratio = 0.2
            random_state = 42
            logger.warning("configs/setup.yaml not found, using defaults")

        # Perform stratified split
        from sklearn.model_selection import train_test_split

        # Get stratification labels
        missing_cols = [col for col in stratify_by if col not in adata.obs.columns]
        if missing_cols:
            raise ValueError(
                f"Stratification columns not found in data: {missing_cols}"
            )

        # Create composite stratification labels if multiple columns
        if len(stratify_by) == 1:
            labels = adata.obs[stratify_by[0]].values
        else:
            # Combine multiple columns for stratification
            labels = (
                adata.obs[stratify_by]
                .apply(lambda x: "_".join(x.astype(str)), axis=1)
                .values
            )
        indices = np.arange(len(labels))

        # Calculate test_size based on split_size
        total_samples = len(adata)
        if split_size >= total_samples:
            logger.warning(
                f"Split size {split_size} >= total samples {total_samples}, using fallback ratio {fallback_ratio}"
            )
            test_size = fallback_ratio
        else:
            test_size = split_size / total_samples
            logger.info(
                f"Using test_size: {test_size:.4f} ({split_size}/{total_samples})"
            )

        # Perform stratified split
        train_idx, eval_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=random_state
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
        encoding_var = self.config.data.target_variable
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

        # Verify batch-corrected files if they exist
        batch_original_file = data_dir / self._generate_filename(
            tissue, "original", batch=True
        )
        batch_train_file = data_dir / self._generate_filename(
            tissue, "train", batch=True
        )
        batch_eval_file = data_dir / self._generate_filename(tissue, "eval", batch=True)

        batch_files_exist = all(
            [
                batch_original_file.exists(),
                batch_train_file.exists(),
                batch_eval_file.exists(),
            ]
        )

        if batch_files_exist:
            logger.info("Verifying batch-corrected files consistency...")

            # Load batch-corrected data
            adata_batch_original = anndata.read_h5ad(batch_original_file)
            adata_batch_train = anndata.read_h5ad(batch_train_file)
            adata_batch_eval = anndata.read_h5ad(batch_eval_file)

            # Verify same number of cells
            if len(adata_batch_original) != len(adata_original):
                raise ValueError(
                    f"Batch-corrected original has different cell count: {len(adata_batch_original)} vs {len(adata_original)}"
                )
            if len(adata_batch_train) != len(adata_train):
                raise ValueError(
                    f"Batch-corrected train has different cell count: {len(adata_batch_train)} vs {len(adata_train)}"
                )
            if len(adata_batch_eval) != len(adata_eval):
                raise ValueError(
                    f"Batch-corrected eval has different cell count: {len(adata_batch_eval)} vs {len(adata_eval)}"
                )

            # Verify same number of genes
            if adata_batch_original.n_vars != adata_original.n_vars:
                raise ValueError(
                    f"Batch-corrected original has different gene count: {adata_batch_original.n_vars} vs {adata_original.n_vars}"
                )
            if adata_batch_train.n_vars != adata_train.n_vars:
                raise ValueError(
                    f"Batch-corrected train has different gene count: {adata_batch_train.n_vars} vs {adata_train.n_vars}"
                )
            if adata_batch_eval.n_vars != adata_eval.n_vars:
                raise ValueError(
                    f"Batch-corrected eval has different gene count: {adata_batch_eval.n_vars} vs {adata_eval.n_vars}"
                )

            # Verify same cell order (obs_names)
            if not adata_batch_original.obs_names.equals(adata_original.obs_names):
                raise ValueError("Batch-corrected original has different cell order")
            if not adata_batch_train.obs_names.equals(adata_train.obs_names):
                raise ValueError("Batch-corrected train has different cell order")
            if not adata_batch_eval.obs_names.equals(adata_eval.obs_names):
                raise ValueError("Batch-corrected eval has different cell order")

            # Verify same gene order (var_names)
            if not adata_batch_original.var_names.equals(adata_original.var_names):
                raise ValueError("Batch-corrected original has different gene order")
            if not adata_batch_train.var_names.equals(adata_train.var_names):
                raise ValueError("Batch-corrected train has different gene order")
            if not adata_batch_eval.var_names.equals(adata_eval.var_names):
                raise ValueError("Batch-corrected eval has different gene order")

            # Verify metadata consistency (obs columns should be identical)
            for file_type, (batch_data, regular_data) in [
                ("original", (adata_batch_original, adata_original)),
                ("train", (adata_batch_train, adata_train)),
                ("eval", (adata_batch_eval, adata_eval)),
            ]:
                if not set(batch_data.obs.columns) == set(regular_data.obs.columns):
                    logger.warning(
                        f"Batch-corrected {file_type} has different metadata columns"
                    )

                # Check key metadata columns are identical
                for col in ["age", "sex", "genotype"]:
                    if col in regular_data.obs.columns:
                        if not batch_data.obs[col].equals(regular_data.obs[col]):
                            raise ValueError(
                                f"Batch-corrected {file_type} has different {col} values"
                            )

            logger.info(
                "✅ Batch-corrected files verified: same cells, genes, and order!"
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
