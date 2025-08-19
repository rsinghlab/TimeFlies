"""Data loading utilities for TimeFlies project."""

import os
import scanpy as sc
import pandas as pd
from typing import Tuple, List, Any
from anndata import AnnData
from ..utils.path_manager import PathManager


class DataLoader:
    """
    A class to handle data loading operations.

    This class manages the loading of datasets and gene lists from specified file paths
    based on the configuration provided.
    """

    def __init__(self, config: Any):
        """
        Initializes the DataLoader with the given configuration.

        Parameters:
        - config: The configuration object.
        """
        self.config = config

        # Use PathManager to get correct data directory (supports test environments)
        self.path_manager = PathManager(config)
        self.Data_dir = self.path_manager.get_raw_data_dir()

        # Prepare the paths to various data files
        self._prepare_paths()

    def _prepare_paths(self) -> None:
        """
        Prepares file paths for data loading.

        This method constructs the full paths to the h5ad data files and other relevant
        files based on the configuration provided.
        """
        # Path to the main h5ad file (directly in tissue directory from PathManager)
        self.h5ad_file_path = os.path.join(
            self.Data_dir,
            getattr(self.config.file_locations, 'training_file', 'fly_train.h5ad'),
        )

        # Path to the evaluation h5ad file
        self.h5ad_eval_file_path = os.path.join(
            self.Data_dir,
            getattr(self.config.file_locations, 'evaluation_file', 'fly_eval.h5ad'),
        )

        # Path to the original unprocessed h5ad file
        self.h5ad_file_path_original = os.path.join(
            self.Data_dir,
            getattr(self.config.file_locations, 'original_file', 'fly_original.h5ad'),
        )

        # Paths for batch-corrected data (same directory structure)
        self.h5ad_file_path_corrected = os.path.join(
            self.Data_dir,
            getattr(self.config.file_locations.batch_corrected_files, 'train', 'fly_train_batch.h5ad'),
        )
        self.h5ad_eval_file_path_corrected = os.path.join(
            self.Data_dir,
            getattr(self.config.file_locations.batch_corrected_files, 'eval', 'fly_eval_batch.h5ad'),
        )

    def load_data(self) -> Tuple[AnnData, AnnData, AnnData]:
        """
        Loads the main datasets.

        This method loads the main, evaluation, and original datasets from their respective
        h5ad files.

        Returns:
        - tuple: A tuple containing AnnData objects for the main, evaluation, and original datasets.
        """
        # Load the main dataset
        adata = sc.read_h5ad(self.h5ad_file_path)

        # Load the evaluation dataset
        adata_eval = sc.read_h5ad(self.h5ad_eval_file_path)

        # Load the original unprocessed dataset
        adata_original = sc.read_h5ad(self.h5ad_file_path_original)

        return adata, adata_eval, adata_original

    def load_corrected_data(self) -> Tuple[AnnData, AnnData]:
        """
        Loads the batch-corrected datasets.

        This method loads the batch-corrected data and evaluation datasets from their
        respective h5ad files.

        Returns:
        - tuple: A tuple containing AnnData objects for the batch-corrected and
                 batch-corrected evaluation datasets.
        """
        # Load the batch-corrected dataset
        adata_corrected = sc.read_h5ad(self.h5ad_file_path_corrected)

        # Load the batch-corrected evaluation dataset
        adata_eval_corrected = sc.read_h5ad(self.h5ad_eval_file_path_corrected)

        return adata_corrected, adata_eval_corrected

    def load_gene_lists(self) -> Tuple[List[str], List[str]]:
        """
        Loads gene lists from CSV files.

        This method loads lists of autosomal and sex-linked genes from CSV files.

        Returns:
        - tuple: A tuple containing two lists, one for autosomal genes and one for sex-linked genes.
        """
        # Use PathManager to get project root and construct gene lists path
        project_root = self.path_manager._get_project_root()
        gene_lists_dir = project_root / "data" / "raw" / "gene_lists"
        
        # Load autosomal gene list from a CSV file
        autosomal_genes = (
            pd.read_csv(
                gene_lists_dir / "autosomal.csv",
                header=None,
                dtype=str,
            )
            .iloc[:, 0]
            .tolist()
        )

        # Load sex-linked gene list from a CSV file
        sex_genes = (
            pd.read_csv(gene_lists_dir / "sex.csv", header=None)
            .iloc[:, 0]
            .tolist()
        )

        return autosomal_genes, sex_genes