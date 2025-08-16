"""Data loading utilities for TimeFlies project."""

import os
import scanpy as sc
import pandas as pd
from typing import Tuple, List, Any
from anndata import AnnData


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

        # Get the directory of the current script
        self.Code_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the data directory based on the configuration
        self.Data_dir = os.path.join(
            self.Code_dir,
            "..", "..", "..",
            "Data",
            "h5ad",
        )

        # Prepare the paths to various data files
        self._prepare_paths()

    def _prepare_paths(self) -> None:
        """
        Prepares file paths for data loading.

        This method constructs the full paths to the h5ad data files and other relevant
        files based on the configuration provided.
        """
        # Get tissue from config
        tissue = self.config.DataParameters.GeneralSettings.tissue

        # Path to the main h5ad file
        self.h5ad_file_path = os.path.join(
            self.Data_dir,
            tissue,
            "uncorrected",
            self.config.FileLocations.training_file,
        )

        # Path to the evaluation h5ad file
        self.h5ad_eval_file_path = os.path.join(
            self.Data_dir,
            tissue,
            "uncorrected",
            self.config.FileLocations.evaluation_file,
        )

        # Path to the original unprocessed h5ad file
        self.h5ad_file_path_original = os.path.join(
            self.Data_dir,
            tissue,
            "uncorrected",
            self.config.FileLocations.original_file,
        )

        # Paths for batch-corrected data and evaluation files
        self.h5ad_file_path_corrected = os.path.join(
            self.Data_dir,
            tissue,
            "batch_corrected",
            self.config.FileLocations.batch_corrected_files.train,
        )
        self.h5ad_eval_file_path_corrected = os.path.join(
            self.Data_dir,
            tissue,
            "batch_corrected",
            self.config.FileLocations.batch_corrected_files.eval,
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
        # Load autosomal gene list from a CSV file
        autosomal_genes = (
            pd.read_csv(
                os.path.join(self.Data_dir, "..", "gene_lists", "autosomal.csv"),
                header=None,
                dtype=str,
            )
            .iloc[:, 0]
            .tolist()
        )

        # Load sex-linked gene list from a CSV file
        sex_genes = (
            pd.read_csv(os.path.join(self.Data_dir, "..", "gene_lists", "sex.csv"), header=None)
            .iloc[:, 0]
            .tolist()
        )

        return autosomal_genes, sex_genes