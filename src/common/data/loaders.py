"""Data loading utilities for TimeFlies project."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import scanpy as sc
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
        # Use standard path logic based on project naming convention
        project_name = getattr(self.config, "project", "fruitfly_aging")
        if project_name == "fruitfly_aging":
            project_for_files = "aging"
        elif project_name == "fruitfly_alzheimers":
            project_for_files = "alzheimers"
        else:
            project_for_files = project_name

        # Standard path templates - users follow this naming convention
        # Get config values
        tissue = getattr(self.config.data, "tissue", "head")
        species = getattr(self.config.data, "species", "drosophila")

        # Standard path templates - users follow this naming convention
        train_filename = f"{species}_{tissue}_{project_for_files}_train.h5ad"
        self.h5ad_file_path = os.path.join(self.Data_dir, train_filename)

        # Path to the evaluation h5ad file
        eval_filename = f"{species}_{tissue}_{project_for_files}_eval.h5ad"
        self.h5ad_eval_file_path = os.path.join(self.Data_dir, eval_filename)

        # Path to the original unprocessed h5ad file
        original_filename = f"{species}_{tissue}_{project_for_files}_original.h5ad"
        self.h5ad_file_path_original = os.path.join(self.Data_dir, original_filename)

        # Paths for batch-corrected data (same directory structure)
        train_batch_filename = (
            f"{species}_{tissue}_{project_for_files}_train_batch.h5ad"
        )
        self.h5ad_file_path_corrected = os.path.join(
            self.Data_dir, train_batch_filename
        )

        eval_batch_filename = f"{species}_{tissue}_{project_for_files}_eval_batch.h5ad"
        self.h5ad_eval_file_path_corrected = os.path.join(
            self.Data_dir, eval_batch_filename
        )

    def load_data(self) -> tuple[AnnData, AnnData, AnnData]:
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

    def load_corrected_data(self) -> tuple[AnnData, AnnData]:
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

    def load_gene_lists(self) -> tuple[list[str], list[str]]:
        """
        Loads gene lists from CSV files using new project structure.

        This method loads lists of autosomal and sex-linked genes from CSV files
        located in the same directory as the data files. If gene files are missing,
        returns empty lists and logs a warning.

        Returns:
        - tuple: A tuple containing two lists, one for autosomal genes and one for sex-linked genes.
                 Returns empty lists if gene files are not found.
        """
        # Gene lists are now in the same directory as the data files
        gene_lists_dir = Path(self.Data_dir)

        autosomal_genes = []
        sex_genes = []

        # Try to load autosomal gene list from a CSV file
        autosomal_path = gene_lists_dir / "autosomal_genes.csv"
        try:
            autosomal_genes = (
                pd.read_csv(autosomal_path, header=None, dtype=str).iloc[:, 0].tolist()
            )
        except FileNotFoundError:
            pass  # Silently skip missing autosomal genes

        # Try to load sex-linked gene list from a CSV file
        sex_path = gene_lists_dir / "sex_genes.csv"
        try:
            sex_genes = pd.read_csv(sex_path, header=None).iloc[:, 0].tolist()
        except FileNotFoundError:
            pass  # Silently skip missing sex genes

        # Log status of gene filtering capability
        if not autosomal_genes and not sex_genes:
            print("⚠ Gene filtering disabled (no reference data found)")

        return autosomal_genes, sex_genes
