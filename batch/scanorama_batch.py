# batch.py
import os
import time
import logging
from typing import List

import numpy as np
import scipy
import scanpy as sc
import scanorama
import matplotlib.pyplot as plt
from Code.config import config

# Configure logging for better control over output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    A class to handle batch correction, data optimization, and visualization for single-cell RNA-seq data.
    """

    def __init__(self, config):
        """
        Initializes the BatchProcessor with the provided configuration.

        Args:
            config (ConfigHandler): Configuration object containing settings and parameters.
        """
        self.config = config
        self.code_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = self._get_data_dir()

        # Validate necessary configuration entries
        self._validate_config()

    def _validate_config(self):
        """
        Validates that necessary configuration entries are present.
        """
        if not self.config.FileLocations.original_file:
            raise ValueError("Original file not specified in configuration.")

        if "original" not in self.config.FileLocations.batch_corrected_files:
            raise ValueError(
                "Batch-corrected original file not specified in configuration."
            )

    def _get_data_dir(self, batch_corrected: bool = False) -> str:
        """
        Constructs the data directory path based on the configuration.

        Args:
            batch_corrected (bool): Whether to use the batch_corrected subdirectory.

        Returns:
            str: The constructed data directory path.
        """
        base_dir = os.path.join(
            self.code_dir,
            "Data",
            "h5ad",
        )
        if batch_corrected:
            return os.path.join(base_dir, "head", "batch_corrected")
        return os.path.join(base_dir, "head", "uncorrected")

    def run_umap(
        self,
        file_name: str,
        title: str,
        save_title: str,
        batch_corrected: bool = False,
        n_comps: int = None,
        n_pcs: int = None,
        color: str = None,
    ) -> None:
        """
        Generates and saves a UMAP plot for the given dataset.
        """

        data_path = os.path.join(self._get_data_dir(batch_corrected), file_name)
        if batch_corrected:
            umap_config = self.config.Setup.batch.umap.batch_corrected
        else:
            umap_config = self.config.Setup.batch.umap.original

        # If UMAP is disabled for this type, skip
        if not umap_config.enabled:
            logger.info(
                f"UMAP generation for {'batch-corrected' if batch_corrected else 'original'} data is disabled in the configuration."
            )
            return

        n_comps = n_comps if n_comps is not None else umap_config.n_comps
        n_pcs = n_pcs if n_pcs is not None else umap_config.n_pcs
        color = color if color is not None else umap_config.color

        # Ensure n_pcs does not exceed n_comps
        if n_pcs and n_comps and n_pcs > n_comps:
            logger.warning(
                "n_pcs is greater than n_comps. Adjusting n_pcs to be equal to n_comps."
            )
            n_pcs = n_comps

        try:
            adata = sc.read_h5ad(data_path)
            adata = adata[adata.obs["sex"] != "mix"].copy()
            adata_vis = adata.copy()

            sc.settings.verbosity = (
                3 if self.config.DataProcessing.Preprocessing.required else 2
            )
            sc.pp.pca(adata_vis, n_comps=n_comps)
            sc.pp.neighbors(adata_vis, n_pcs=n_pcs)
            sc.tl.umap(adata_vis)

            # Create UMAP output directory if it doesn't exist
            umap_output_dir = os.path.join(
                self.code_dir,
                "Analysis",
                "umaps",
            )
            os.makedirs(umap_output_dir, exist_ok=True)

            # Generate UMAP plot without saving via Scanpy
            sc.pl.umap(
                adata_vis,
                color=color,
                show=False,
            )

            # Customize plot before saving
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title="Age (Days)")  # Dynamic legend title

            # Construct the full path to save the plot
            umap_file_path = os.path.join(umap_output_dir, f"{save_title}.png")

            plt.title(title, fontsize=12, weight="bold")
            plt.xlabel("UMAP1", fontsize=8)
            plt.ylabel("UMAP2", fontsize=8)
            plt.savefig(umap_file_path, dpi=300)  # Increased DPI for better resolution
            plt.close()
            logger.info(f"UMAP plot saved as {umap_file_path}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
        except KeyError as e:
            logger.error(f"Missing expected key in data: {e}")
        except Exception as e:
            logger.error(f"Failed to generate UMAP plot: {e}")

    def optimize_and_save_adata(self, filename: str) -> None:
        """
        Optimizes an AnnData object by dropping specified columns, converting data types for memory efficiency,
        and saving the optimized data.

        Args:
            filename (str): Name of the .h5ad file to optimize and save.
        """
        try:
            # Determine if the file is batch-corrected based on config
            batch_corrected = (
                filename
                == self.config.FileLocations.batch_corrected_files.get("original")
            )

            if not batch_corrected:
                logger.info(
                    f"Skipping optimization for non-batch-corrected file: {filename}"
                )
                return

            data_path = os.path.join(
                self._get_data_dir(batch_corrected=batch_corrected), filename
            )
            adata = sc.read_h5ad(data_path)

            columns_to_drop = self.config.Setup.batch.columns_to_drop
            adata.obs.drop(
                columns=columns_to_drop, axis=1, inplace=True, errors="ignore"
            )

            # Convert float64 to float32 in adata.var and adata.obs
            for df in [adata.var, adata.obs]:
                float_cols = df.select_dtypes(include=["float64"]).columns
                if not float_cols.empty:
                    df[float_cols] = df[float_cols].astype("float32")

            # Convert the main matrix to float32
            if isinstance(adata.X, np.ndarray):
                adata.X = adata.X.astype("float32")
            elif scipy.sparse.issparse(adata.X):
                adata.X = adata.X.astype("float32")

            # Ensure the save directory exists
            save_dir = self._get_data_dir(batch_corrected=batch_corrected)
            os.makedirs(save_dir, exist_ok=True)

            # Determine the new filename
            new_filename = filename.replace(".h5ad", "_optimized.h5ad")
            optimized_file_path = os.path.join(save_dir, new_filename)
            adata.write(optimized_file_path)
            logger.info(f"Optimized AnnData saved to {optimized_file_path}")
        except Exception as e:
            logger.error(f"Failed to optimize and save AnnData: {e}")

    def print_adata_summary(self, filename: str, batch: bool = False) -> None:
        """
        Prints a summary of an AnnData object, including shapes, types, memory usage, sparsity, and first 5 rows of obs.

        Args:
            filename (str): Name of the .h5ad file to summarize.
            batch (bool): Whether the data is batch corrected.
        """
        try:
            data_path = os.path.join(
                self._get_data_dir(batch_corrected=batch), filename
            )
            adata = sc.read_h5ad(data_path)

            logger.info(f"--- Summary for {filename} ---")
            logger.info(f"Main matrix (X) shape: {adata.X.shape}")
            logger.info(f"Data type of X: {type(adata.X)}")
            logger.info(f"Data dtype of X: {adata.X.dtype}")

            logger.info(f"Observations (obs) DataFrame shape: {adata.obs.shape}")
            logger.info(
                f"Observations memory usage:\n{adata.obs.memory_usage(deep=True)}"
            )
            logger.info(f"First 5 rows of obs DataFrame:\n{adata.obs.head()}")

            logger.info(f"Variables (var) DataFrame shape: {adata.var.shape}")
            logger.info(f"Variables memory usage:\n{adata.var.memory_usage(deep=True)}")
            logger.debug(f"Variables DataFrame:\n{adata.var}")

            sparsity_percentage = self._estimate_sparsity(adata.X)
            logger.info(f"Data Sparsity: {sparsity_percentage:.2f}%")
        except Exception as e:
            logger.error(f"Failed to print AnnData summary: {e}")

    @staticmethod
    def _estimate_sparsity(matrix) -> float:
        """
        Estimates the sparsity of a matrix.

        Args:
            matrix (np.ndarray or scipy.sparse matrix): The data matrix.

        Returns:
            float: Sparsity percentage.
        """
        if scipy.sparse.issparse(matrix):
            total_elements = matrix.shape[0] * matrix.shape[1]
            non_zero_elements = matrix.nnz
        else:
            total_elements = matrix.size
            non_zero_elements = np.count_nonzero(matrix)

        sparsity = ((total_elements - non_zero_elements) / total_elements) * 100
        return sparsity

    def remove_last_suffix_from_obs_names(self, adata, separator: str = None) -> None:
        """
        Removes the last suffix from cell identifiers in an AnnData object.

        Args:
            adata (AnnData): The AnnData object to modify.
            separator (str, optional): The character used to separate parts of the identifiers.
                                        Defaults to config.Setup.batch.separator.
        """
        separator = (
            separator if separator is not None else self.config.Setup.batch.separator
        )
        try:
            updated_obs_names = [
                (
                    separator.join(name.split(separator)[:-1])
                    if separator in name
                    else name
                )
                for name in adata.obs_names
            ]
            adata.obs_names = updated_obs_names
            logger.info("Removed last suffix from observation names.")
        except Exception as e:
            logger.error(f"Failed to remove last suffix from obs names: {e}")

    @staticmethod
    def extract_age_from_sex_age(sex_age_str: str) -> int:
        """
        Extracts the age component from a combined sex_age string.

        Args:
            sex_age_str (str): The combined sex and age string.

        Returns:
            int: The extracted age.
        """
        try:
            return int(sex_age_str.split("_")[1])
        except (IndexError, ValueError) as e:
            logger.error(f"Failed to extract age from '{sex_age_str}': {e}")
            return -1  # Return a default or error value

    def process_or_load_data(self, adata, name: str) -> sc.AnnData:
        """
        Processes and corrects an AnnData object if not already done, otherwise loads the processed data.

        Args:
            adata (AnnData): The AnnData object to process.
            name (str): Identifier for the data (e.g., 'original').

        Returns:
            AnnData: The processed or loaded AnnData object.
        """
        try:
            suffix = self.config.Setup.batch.corrected_suffix  # e.g., "_batch"
            corrected_filename = f"{name}{suffix}.h5ad"

            corrected_file_path = os.path.join(
                self._get_data_dir(batch_corrected=True), corrected_filename
            )

            if os.path.exists(corrected_file_path):
                logger.info(
                    f"Loading existing batch-corrected data from {corrected_file_path}"
                )
                return sc.read_h5ad(corrected_file_path)

            # Split data by age and correct
            adata_list_by_age = [
                adata[adata.obs["age"] == age].copy()
                for age in adata.obs["age"].unique()
            ]
            corrected_adatas = scanorama.correct_scanpy(adata_list_by_age)
            corrected_adata = corrected_adatas[0].concatenate(
                corrected_adatas[1:], batch_key="age"
            )

            os.makedirs(os.path.dirname(corrected_file_path), exist_ok=True)
            corrected_adata.write(corrected_file_path)
            logger.info(f"Batch-corrected data saved to {corrected_file_path}")

            return corrected_adata
        except Exception as e:
            logger.error(f"Failed to process or load data: {e}")
            raise

    def validate_and_reorder_samples(
        self, combined_processed_adata: sc.AnnData, original_order: List[str]
    ) -> sc.AnnData:
        """
        Validates and reorders samples in an AnnData object based on the original order.

        Args:
            combined_processed_adata (AnnData): The processed AnnData object.
            original_order (List[str]): Original sample order.

        Returns:
            AnnData: The validated and reordered AnnData object.
        """
        try:
            all_samples_present = all(
                sample in combined_processed_adata.obs_names
                for sample in original_order
            )
            extra_samples = set(combined_processed_adata.obs_names) - set(
                original_order
            )

            if all_samples_present and not extra_samples:
                logger.info(
                    "All original samples are present and no extra samples found. Proceeding with reordering."
                )
            else:
                if not all_samples_present:
                    logger.warning(
                        "Not all original samples are present in the processed data."
                    )
                if extra_samples:
                    logger.warning(
                        "Extra samples found in the processed data that are not in the original order."
                    )

            if all_samples_present and not extra_samples:
                combined_processed_adata = combined_processed_adata[original_order, :]

                # Create UMAP output directory if it doesn't exist
                umap_output_dir = os.path.join(
                    self._get_data_dir(batch_corrected=True),
                    self.config.Setup.batch.umap.batch_corrected.output_dir,
                )
                os.makedirs(umap_output_dir, exist_ok=True)

                # Retrieve the corrected filename for 'original'
                corrected_filename = (
                    self.config.FileLocations.batch_corrected_files.get("original")
                )
                if not corrected_filename:
                    raise ValueError(
                        "No batch-corrected filename found for 'original' in configuration."
                    )

                corrected_file_path = os.path.join(
                    self._get_data_dir(batch_corrected=True), corrected_filename
                )
                combined_processed_adata.write(corrected_file_path)
                logger.info(
                    f"Reordered and validated AnnData saved to {corrected_file_path}"
                )

            return combined_processed_adata
        except Exception as e:
            logger.error(f"Failed to validate and reorder samples: {e}")
            raise

    def perform_eda(self) -> None:
        """
        Performs detailed EDA on batch-corrected and/or original .h5ad files based on configuration.
        """
        try:
            # Fetch EDA configuration flags
            perform_eda_batch = self.config.Setup.batch.perform_eda.get(
                "batch_corrected", False
            )
            perform_eda_original = self.config.Setup.batch.perform_eda.get(
                "original", False
            )

            # Perform EDA on batch-corrected original file if enabled
            if perform_eda_batch:
                batch_corrected_original_file = (
                    self.config.FileLocations.batch_corrected_files.get("original")
                )
                if batch_corrected_original_file:
                    logger.info("Starting EDA on batch-corrected original file...")
                    logger.info(
                        f"\n--- Summary for {batch_corrected_original_file} ---"
                    )
                    self.print_adata_summary(batch_corrected_original_file, batch=True)

                    # Additional analyses
                    adata = sc.read_h5ad(
                        os.path.join(
                            self._get_data_dir(batch_corrected=True),
                            batch_corrected_original_file,
                        )
                    )

                    # Total number of genes
                    total_genes = adata.n_vars
                    logger.info(f"Total number of genes: {total_genes}")

                    # Total number of cells
                    total_cells = adata.n_obs
                    logger.info(f"Total number of cells: {total_cells}")

                    # Average mitochondrial gene percentage, if available
                    if "pct_counts_mt" in adata.obs.columns:
                        avg_pct_mt = adata.obs["pct_counts_mt"].mean()
                        logger.info(
                            f"Average mitochondrial gene percentage: {avg_pct_mt:.2f}%"
                        )

                    # Average number of genes detected per cell, if available
                    if "n_genes_by_counts" in adata.obs.columns:
                        avg_n_genes = adata.obs["n_genes_by_counts"].mean()
                        logger.info(
                            f"Average number of genes detected per cell: {avg_n_genes:.2f}"
                        )

                    # Example: Distribution of total counts
                    if "total_counts" in adata.obs.columns:
                        median_total_counts = adata.obs["total_counts"].median()
                        logger.info(
                            f"Median total counts per cell: {median_total_counts}"
                        )
                else:
                    logger.warning(
                        "Batch-corrected original file not found in configuration."
                    )

            else:
                logger.info(
                    "EDA on batch-corrected original file is disabled in the configuration."
                )

            # Perform EDA on the original file if enabled
            if perform_eda_original:
                original_file = self.config.FileLocations.original_file
                if original_file:
                    logger.info(f"\n--- Summary for {original_file} ---")
                    self.print_adata_summary(original_file, batch=False)

                    # Additional analyses
                    adata_original = sc.read_h5ad(
                        os.path.join(
                            self._get_data_dir(batch_corrected=False), original_file
                        )
                    )

                    # Total number of genes
                    total_genes_original = adata_original.n_vars
                    logger.info(f"Total number of genes: {total_genes_original}")

                    # Total number of cells
                    total_cells_original = adata_original.n_obs
                    logger.info(f"Total number of cells: {total_cells_original}")

                    # Average mitochondrial gene percentage, if available
                    if "pct_counts_mt" in adata_original.obs.columns:
                        avg_pct_mt_original = adata_original.obs["pct_counts_mt"].mean()
                        logger.info(
                            f"Average mitochondrial gene percentage: {avg_pct_mt_original:.2f}%"
                        )

                    # Average number of genes detected per cell, if available
                    if "n_genes_by_counts" in adata_original.obs.columns:
                        avg_n_genes_original = adata_original.obs[
                            "n_genes_by_counts"
                        ].mean()
                        logger.info(
                            f"Average number of genes detected per cell: {avg_n_genes_original:.2f}"
                        )

                    # Example: Distribution of total counts
                    if "total_counts" in adata_original.obs.columns:
                        median_total_counts_original = adata_original.obs[
                            "total_counts"
                        ].median()
                        logger.info(
                            f"Median total counts per cell: {median_total_counts_original}"
                        )
                else:
                    logger.warning("Original file not found in configuration.")

            else:
                logger.info(
                    "EDA on the original file is disabled in the configuration."
                )

        except Exception as e:
            logger.error(f"Failed to perform detailed EDA: {e}")
            raise

    def run(
        self,
        batch_correction: bool = None,
        umap_batch: bool = False,
        umap_original: bool = False,
        optimize: bool = False,
    ) -> None:
        """
        Executes the batch processing pipeline based on the provided configuration.

        Args:
            batch_correction (bool, optional): Whether to perform batch correction.
                                            Defaults to config.Setup.batch.enabled.
            umap_batch (bool, optional): Whether to generate UMAP for batch-corrected data.
                                        Defaults to config.Setup.batch.umap.batch_corrected.enabled.
            umap_original (bool, optional): Whether to generate UMAP for original data.
                                        Defaults to config.Setup.batch.umap.original.enabled.
            optimize (bool, optional): Whether to optimize and save the batch-corrected data.
                                        Defaults to config.Setup.batch.optimize.
        """
        # Set defaults from config if not provided
        if batch_correction is None:
            batch_correction = self.config.Setup.batch.enabled
        if not umap_batch:
            umap_batch = self.config.Setup.batch.umap.batch_corrected.enabled
        if not umap_original:
            umap_original = self.config.Setup.batch.umap.original.enabled
        if not optimize:
            optimize = self.config.Setup.batch.optimize

        start_time = time.time()

        try:
            original_file = self.config.FileLocations.original_file
            original_path = os.path.join(self._get_data_dir(), original_file)
            adata = sc.read_h5ad(original_path)

            if batch_correction:
                # Select highly variable genes if enabled
                if self.config.GenePreprocessing.GeneFiltering.highly_variable_genes:
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.highly_variable_genes(
                        adata, n_top_genes=self.config.Setup.batch.n_top_genes
                    )
                    highly_variable_genes = adata.var.highly_variable
                    adata = adata[:, highly_variable_genes]
                    logger.info(
                        f"Selected top {self.config.Setup.batch.n_top_genes} highly variable genes."
                    )

                original_order = adata.obs_names.tolist()

                # Batch correction
                processed_data = self.process_or_load_data(adata, "original")
                self.remove_last_suffix_from_obs_names(processed_data)
                processed_data.obs["age"] = processed_data.obs["sex_age"].apply(
                    self.extract_age_from_sex_age
                )
                processed_data = self.validate_and_reorder_samples(
                    processed_data, original_order
                )

                # Assertions to ensure data integrity
                assert len(processed_data) == len(
                    original_order
                ), "Sample count mismatch."
                assert all(
                    processed_data.obs_names == original_order
                ), "Sample order mismatch."

                # Perform EDA
                self.perform_eda()

                # Optimize data if requested
                if optimize:
                    batch_corrected_original_file = (
                        self.config.FileLocations.batch_corrected_files.get("original")
                    )
                    if not batch_corrected_original_file:
                        logger.error(
                            "No batch-corrected filename found for 'original' in configuration."
                        )
                    else:
                        self.optimize_and_save_adata(batch_corrected_original_file)

            # Allow optimization even if batch correction is disabled
            if optimize and not batch_correction:
                batch_corrected_original_file = (
                    self.config.FileLocations.batch_corrected_files.get("original")
                )
                if batch_corrected_original_file:
                    self.optimize_and_save_adata(batch_corrected_original_file)
                else:
                    logger.warning(
                        "Optimization is enabled, but no batch-corrected original file found in configuration."
                    )

            else:
                if optimize and batch_correction:
                    # Already optimized in the batch_correction block
                    pass
                elif optimize:
                    # Handled above
                    pass

            # Perform EDA regardless of batch correction to ensure summaries are available
            if not batch_correction:
                self.perform_eda()

            # Generate UMAP for batch-corrected data if requested
            if umap_batch:
                batch_corrected_original_file = (
                    self.config.FileLocations.batch_corrected_files.get("original")
                )
                if not batch_corrected_original_file:
                    logger.error(
                        "No batch-corrected filename found for 'original' in configuration."
                    )
                else:
                    self.run_umap(
                        file_name=batch_corrected_original_file,
                        title="UMAP of Fly Data After Correction",
                        save_title="UMAP_after_correction",  # Removed "UMAP_" prefix
                        batch_corrected=True,
                    )

            # Generate UMAP for original data if requested
            if umap_original:
                original_file = self.config.FileLocations.original_file
                if not original_file:
                    logger.error("No original filename found in configuration.")
                else:
                    self.run_umap(
                        file_name=original_file,
                        title="UMAP of Fly Data Before Correction",
                        save_title="UMAP_before_correction",  # Removed "UMAP_" prefix
                        batch_corrected=False,
                    )

        except AssertionError as ae:
            logger.error(f"Assertion failed: {ae}")
        except Exception as e:
            logger.error(f"An error occurred during the run: {e}")
        finally:
            end_time = time.time()
            duration_seconds = end_time - start_time
            if duration_seconds < 60:
                logger.info(f"The task took {round(duration_seconds)} seconds.")
            else:
                minutes = int(duration_seconds // 60)
                seconds = round(duration_seconds % 60)
                logger.info(f"The task took {minutes} minutes and {seconds} seconds.")


if __name__ == "__main__":
    processor = BatchProcessor(config)

    # Use config settings to control batch correction, UMAP generation, and optimization
    processor.run(
        batch_correction=config.Setup.batch.enabled,  # Overall batch correction setting
        umap_batch=config.Setup.batch.umap.batch_corrected.enabled,  # UMAP for batch-corrected data
        umap_original=config.Setup.batch.umap.original.enabled,  # UMAP for original data
        optimize=config.Setup.batch.optimize,  # Optimization setting
    )
