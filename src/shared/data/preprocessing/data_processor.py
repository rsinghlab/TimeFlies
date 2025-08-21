"""Main data preprocessing utilities for TimeFlies project."""

import os
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore
import dill as pickle
import logging
from typing import Tuple, Optional, List, Any
from anndata import AnnData

from ...utils.path_manager import PathManager

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle data preprocessing.

    This class preprocesses the input data based on the configuration settings or loads
    preprocessed data if no further processing is required.
    """

    def __init__(self, config: Any, adata: AnnData, adata_corrected: AnnData):
        """
        Initializes the DataPreprocessor with the given configuration and datasets.

        Parameters:
        - config: The configuration object containing all necessary parameters.
        - adata: The main AnnData object containing the dataset.
        - adata_corrected: The batch-corrected AnnData object.
        """
        self.config = config
        self.adata = adata
        self.adata_corrected = adata_corrected

        # Initialize PathManager
        self.path_manager = PathManager(config)

    def process_adata(self, adata: AnnData) -> AnnData:
        """
        Process an AnnData object based on the provided configuration.

        Parameters:
        - adata: The AnnData object to be processed.

        Returns:
        - adata: The processed AnnData object.
        """
        config = self.config

        # Include or exclude 'mix' sex
        include_mix = getattr(config.data.filtering, "include_mixed_sex", False)
        if not include_mix:
            adata = adata[adata.obs["sex"] != "mix"].copy()

        # Filter based on 'sex_type' if specified
        sex_type = getattr(config.data, "sex_type", "all").lower()
        if sex_type in ["male", "female"]:
            adata = adata[adata.obs["sex"] == sex_type].copy()

        # Filter based on 'cell_type' if specified
        cell_type = getattr(config.data, "cell_type", "all")
        if cell_type != "all":
            adata = adata[adata.obs["afca_annotation_broad"] == cell_type].copy()

        # Shuffle genes if required
        shuffle_genes = getattr(
            config.preprocessing.shuffling, "shuffle_genes", False
        )
        global_random_state = getattr(config.general, "random_state", 42)
        if shuffle_genes:
            # Use global random state for consistency
            rng = np.random.default_rng(global_random_state)
            gene_order = rng.permutation(adata.var_names)
            adata = adata[:, gene_order]

        # Sample data if required (only for random splits - genotype splits sample after splitting)
        split_method = getattr(config.data.split, "method", "random").lower()
        num_samples = getattr(config.data.sampling, "samples", None)
        
        if num_samples and num_samples < adata.n_obs and split_method == "random":
            random_state = getattr(config.general, "random_state", 42)
            sample_indices = adata.obs.sample(
                n=num_samples, random_state=random_state
            ).index
            adata = adata[sample_indices, :]

        # Select variables (genes) if required
        num_variables = getattr(config.data.sampling, "variables", None)
        if num_variables and num_variables < adata.n_vars:
            adata = adata[:, adata.var_names[:num_variables]]

        return adata

    def split_data(self, dataset: AnnData, evaluation_mode: bool = False) -> Tuple[AnnData, AnnData]:
        """
        Split the dataset into training and testing subsets based on the configuration.

        Parameters:
        - dataset: The dataset to be split.

        Returns:
        - train_subset: The training subset.
        - test_subset: The testing subset.
        """
        config = self.config
        split_method = getattr(config.data.split, "method", "random").lower()

        if split_method == "sex":
            train_sex = getattr(
                config.data.split.sex, "train", "male"
            ).lower()
            test_sex = getattr(
                config.data.train_test_split.test, "sex", "female"
            ).lower()

            train_subset = dataset[dataset.obs["sex"].str.lower() == train_sex].copy()
            test_subset = dataset[dataset.obs["sex"].str.lower() == test_sex].copy()

            # Apply test_ratio to reduce test set size if specified (for performance)
            test_ratio = getattr(config.data.split.sex, "test_ratio", 1.0)
            if test_ratio < 1.0:
                random_state = getattr(config.general, "random_state", 42)
                _, test_subset = train_test_split(
                    test_subset,
                    test_size=test_ratio,
                    random_state=random_state,
                )
                print(f"Applied test_ratio {test_ratio}: Using {test_subset.n_obs} test cells")
            
            # Apply sampling after sex split to get balanced samples
            num_samples = getattr(config.data.sampling, "samples", None)
            if num_samples:
                random_state = getattr(config.general, "random_state", 42)
                
                # Sample from train set
                if num_samples < train_subset.n_obs:
                    train_sample_indices = train_subset.obs.sample(
                        n=min(num_samples, train_subset.n_obs), 
                        random_state=random_state
                    ).index
                    train_subset = train_subset[train_sample_indices, :]
                
                # Sample from test set
                if num_samples < test_subset.n_obs:
                    test_sample_indices = test_subset.obs.sample(
                        n=min(num_samples, test_subset.n_obs), 
                        random_state=random_state
                    ).index
                    test_subset = test_subset[test_sample_indices, :]

        elif split_method == "tissue":
            train_tissue = getattr(
                config.data.train_test_split.train, "tissue", "head"
            ).lower()
            test_tissue = getattr(
                config.data.train_test_split.test, "tissue", "body"
            ).lower()

            train_subset = dataset[
                dataset.obs["tissue"].str.lower() == train_tissue
            ].copy()
            test_subset = dataset[
                dataset.obs["tissue"].str.lower() == test_tissue
            ].copy()

            # Ensure common genes between train and test data
            common_genes = train_subset.var_names.intersection(test_subset.var_names)
            train_subset = train_subset[:, common_genes]
            test_subset = test_subset[:, common_genes]

            # Apply test_ratio to reduce test set size if specified (for performance)
            test_ratio = getattr(config.data.split.tissue, "test_ratio", 1.0)
            if test_ratio < 1.0:
                random_state = getattr(config.general, "random_state", 42)
                _, test_subset = train_test_split(
                    test_subset,
                    test_size=test_ratio,
                    random_state=random_state,
                )
                print(f"Applied test_ratio {test_ratio}: Using {test_subset.n_obs} test cells")
            
            # Apply sampling after tissue split to get balanced samples
            num_samples = getattr(config.data.sampling, "samples", None)
            if num_samples:
                random_state = getattr(config.general, "random_state", 42)
                
                # Sample from train set
                if num_samples < train_subset.n_obs:
                    train_sample_indices = train_subset.obs.sample(
                        n=min(num_samples, train_subset.n_obs), 
                        random_state=random_state
                    ).index
                    train_subset = train_subset[train_sample_indices, :]
                
                # Sample from test set
                if num_samples < test_subset.n_obs:
                    test_sample_indices = test_subset.obs.sample(
                        n=min(num_samples, test_subset.n_obs), 
                        random_state=random_state
                    ).index
                    test_subset = test_subset[test_sample_indices, :]

        elif split_method == "genotype":
            # Genotype-based splitting for Alzheimer's analysis
            
            # Handle multiple test genotypes (e.g., "alzheimers" maps to AB42, hTau)
            test_genotype_spec = getattr(
                config.data.split.genotype, "test", "alzheimers"
            ).lower()
            
            if test_genotype_spec == "alzheimers":
                # Test on all alzheimer's genotypes
                test_genotypes = ["ab42", "htau"]
                test_subset = dataset[
                    dataset.obs["genotype"].str.lower().isin(test_genotypes)
                ].copy()
            else:
                # Test on specific genotype
                test_subset = dataset[
                    dataset.obs["genotype"].str.lower() == test_genotype_spec
                ].copy()
            
            if evaluation_mode:
                # For evaluation, we only need test data (all alzheimer's flies)
                # Create a dummy train_subset to maintain interface compatibility
                train_subset = test_subset[:0].copy()  # Empty subset with same structure
                
                print(f"Evaluation mode: Using all {test_subset.n_obs} test genotype cells")
            else:
                # For training, create proper train subset (control flies)
                train_genotype = getattr(
                    config.data.split.genotype, "train", "control"
                ).lower()
                
                train_subset = dataset[
                    dataset.obs["genotype"].str.lower() == train_genotype
                ].copy()
            
            # Apply test_ratio to reduce test set size if specified (for performance)
            test_ratio = getattr(config.data.split.genotype, "test_ratio", 1.0)
            if test_ratio < 1.0:
                random_state = getattr(config.general, "random_state", 42)
                _, test_subset = train_test_split(
                    test_subset,
                    test_size=test_ratio,
                    random_state=random_state,
                )
                print(f"Applied test_ratio {test_ratio}: Using {test_subset.n_obs} test cells")
            
            # Apply sampling after genotype split to get balanced samples
            num_samples = getattr(config.data.sampling, "samples", None)
            if num_samples and not evaluation_mode:
                # Only apply sampling during training
                random_state = getattr(config.general, "random_state", 42)
                
                # Sample from train set
                if num_samples < train_subset.n_obs:
                    train_sample_indices = train_subset.obs.sample(
                        n=min(num_samples, train_subset.n_obs), 
                        random_state=random_state
                    ).index
                    train_subset = train_subset[train_sample_indices, :]
                
                # Sample from test set
                if num_samples < test_subset.n_obs:
                    test_sample_indices = test_subset.obs.sample(
                        n=min(num_samples, test_subset.n_obs), 
                        random_state=random_state
                    ).index
                    test_subset = test_subset[test_sample_indices, :]

        else:
            # Perform a stratified train-test split based on encoding variable
            encoding_var = getattr(config.data, "encoding_variable", "age")
            test_size = getattr(config.data.train_test_split, "test_split", 0.2)
            random_state = getattr(config.general, "random_state", 42)

            train_subset, test_subset = train_test_split(
                dataset,
                stratify=dataset.obs[encoding_var],
                test_size=test_size,
                random_state=random_state,
            )

            # Note: Legacy gene selection from corrected data removed - 
            # batch correction now uses all genes

        return train_subset, test_subset

    def select_highly_variable_genes(
        self, train_subset: AnnData, test_subset: AnnData
    ) -> Tuple[AnnData, AnnData, Optional[List[str]]]:
        """
        Select highly variable genes from the training data and apply to both training and testing data.
        Simplified version for specific use cases only.

        Parameters:
        - train_subset: The training subset.
        - test_subset: The testing subset.

        Returns:
        - train_subset: The training subset with selected genes.
        - test_subset: The testing subset with selected genes.
        - highly_variable_genes: List of highly variable genes.
        """
        config = self.config
        highly_variable_genes = None

        if getattr(
            config.preprocessing.genes, "highly_variable_genes", False
        ):
            normal = train_subset.copy()
            sc.pp.normalize_total(normal, target_sum=1e4)
            sc.pp.log1p(normal)
            sc.pp.highly_variable_genes(normal, n_top_genes=5000)

            highly_variable_genes = normal.var_names[
                normal.var.highly_variable
            ].tolist()
            train_subset = train_subset[:, highly_variable_genes]
            test_subset = test_subset[:, highly_variable_genes]

        return train_subset, test_subset, highly_variable_genes

    def prepare_labels(
        self, train_subset: AnnData, test_subset: AnnData
    ) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """
        Prepare labels for training and testing data.

        Parameters:
        - train_subset: The training subset.
        - test_subset: The testing subset.

        Returns:
        - train_labels: One-hot encoded training labels.
        - test_labels: One-hot encoded testing labels.
        - label_encoder: The label encoder used to transform the labels.
        """
        config = self.config
        encoding_var = getattr(config.data, "encoding_variable", "age")
        label_encoder = LabelEncoder()

        train_labels = label_encoder.fit_transform(train_subset.obs[encoding_var])
        test_labels = label_encoder.transform(test_subset.obs[encoding_var])

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        return train_labels, test_labels, label_encoder

    def normalize_data(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], bool]:
        """
        Normalize training and testing data if required.

        Parameters:
        - train_data: Training data.
        - test_data: Testing data.

        Returns:
        - train_data: Normalized training data.
        - test_data: Normalized testing data.
        - scaler: The scaler used for normalization.
        - is_scaler_fit: Flag indicating whether the scaler was fit (i.e., normalization was applied).
        """
        config = self.config
        norm_on = getattr(config.data_processing.normalization, "enabled", False)
        bc_on = getattr(config.data.batch_correction, "enabled", False)

        if norm_on:
            logging.info("Normalizing data...")
            # scVI data is *already* log1p – so skip that part
            if not bc_on:
                train_data = np.log1p(train_data)
                test_data = np.log1p(test_data)

            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            is_scaler_fit = True
            logging.info("Data normalized.")
        else:
            scaler, is_scaler_fit = None, False

        return train_data, test_data, scaler, is_scaler_fit

    def reshape_for_cnn(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape data for CNN models.

        Parameters:
        - train_data: Training data.
        - test_data: Testing data.

        Returns:
        - train_data: Reshaped training data.
        - test_data: Reshaped testing data.
        """
        train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
        test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
        return train_data, test_data

    def create_reference_data(self, train_data: np.ndarray) -> np.ndarray:
        """
        Create reference data by sampling from training data.

        Parameters:
        - train_data: Training data.

        Returns:
        - reference_data: Reference data.
        """
        config = self.config
        reference_size = getattr(config.interpretation.shap, "reference_size", 100)
        if reference_size > train_data.shape[0]:
            reference_size = train_data.shape[0]

        reference_data = train_data[
            np.random.choice(train_data.shape[0], reference_size, replace=False)
        ]
        return reference_data

    def prepare_data(self):
        """
        Preprocesses the data based on the configuration.

        Returns:
        - tuple: A tuple containing preprocessed train data, test data, labels, and other related objects.
        """
        config = self.config

        # Decide which dataset to use based on batch correction setting
        batch_correction_enabled = getattr(
            config.data.batch_correction, "enabled", False
        )

        if batch_correction_enabled and self.adata_corrected is not None:
            try:
                # Only process batch-corrected data if we're using it
                print("Using batch-corrected data...")
                adata_corrected = self.adata_corrected.copy()

                # Replace .X with scVI-normalised expression if it exists
                if "scvi_normalized" in adata_corrected.layers:
                    adata_corrected.X = adata_corrected.layers["scvi_normalized"]
                    print("Using scVI-normalized expression layer")

                # Process the corrected data
                dataset_to_use = self.process_adata(adata_corrected)
            except Exception as e:
                print(f"Warning: Could not use batch-corrected data: {e}")
                print("Falling back to uncorrected data...")
                dataset_to_use = self.process_adata(self.adata)
        else:
            # Only process uncorrected data if we're using it
            if batch_correction_enabled:
                print(
                    "Batch correction requested but batch-corrected data not available"
                )
            print("Using uncorrected data...")
            dataset_to_use = self.process_adata(self.adata)

        # Split the data
        train_subset, test_subset = self.split_data(dataset_to_use)

        # Select highly variable genes if required
        (
            train_subset,
            test_subset,
            highly_variable_genes,
        ) = self.select_highly_variable_genes(train_subset, test_subset)

        # Print data sizes and class counts
        print(f"Training data size: {train_subset.shape}")
        print(f"Testing data size: {test_subset.shape}")

        encoding_var = getattr(config.data, "encoding_variable", "age")
        print("\nCounts of each class in the training data:")
        print(train_subset.obs[encoding_var].value_counts())

        print("\nCounts of each class in the testing data:")
        print(test_subset.obs[encoding_var].value_counts())

        # Prepare labels
        train_labels, test_labels, label_encoder = self.prepare_labels(
            train_subset, test_subset
        )

        # Extract data arrays
        train_data = train_subset.X
        if hasattr(train_data, "toarray"):
            train_data = train_data.toarray()

        test_data = test_subset.X
        if hasattr(test_data, "toarray"):
            test_data = test_data.toarray()

        # Normalize data if required
        train_data, test_data, scaler, is_scaler_fit = self.normalize_data(
            train_data, test_data
        )

        # Reshape data for CNN if required
        model_type = getattr(config.data, "model", "mlp").lower()
        if model_type == "cnn":
            train_data, test_data = self.reshape_for_cnn(train_data, test_data)

        # Create reference data
        reference_data = self.create_reference_data(train_data)

        # Get mix_included flag from config
        mix_included = getattr(config.data.filtering, "include_mixed_sex", False)

        return (
            train_data,
            test_data,
            train_labels,
            test_labels,
            label_encoder,
            reference_data,
            scaler,
            is_scaler_fit,
            highly_variable_genes,
            mix_included,
        )

    def prepare_final_eval_data(
        self,
        adata,
        label_encoder,
        num_features,
        scaler,
        is_scaler_fit,
        highly_variable_genes,
        mix_included,
    ):
        """
        Preprocess the final evaluation data based on the provided configuration parameters.

        This method follows the legacy implementation to ensure consistency.

        Args:
            adata (AnnData): The input evaluation data.
            label_encoder (LabelEncoder): The label encoder used to transform the labels.
            num_features (int): The number of features to use.
            scaler (StandardScaler or None): The scaler used for data normalization.
            is_scaler_fit (bool): Flag indicating if the scaler was fit.
            highly_variable_genes (list): List of highly variable genes.
            mix_included (bool): Flag indicating if 'mix' samples were included during training.

        Returns:
            test_data (ndarray): Testing data.
            test_labels (ndarray): Labels for the testing data.
            label_encoder (LabelEncoder): The label encoder used to transform the labels.
        """
        config = self.config

        batch_correction_enabled = getattr(
            config.data.batch_correction, "enabled", False
        )

        if batch_correction_enabled:
            adata = adata.copy()
            if "scvi_normalized" in adata.layers:
                adata.X = adata.layers["scvi_normalized"]

        # Remove 'mix' if specified
        if mix_included is False:
            adata = adata[adata.obs["sex"] != "mix"].copy()

        # Check if the specified cell type is 'all'
        cell_type = getattr(config.data, "cell_type", "all").lower()
        if cell_type != "all":
            adata = adata[adata.obs["afca_annotation_broad"] == cell_type].copy()

        # Sex Mapping
        sex_type = getattr(config.data, "sex_type", "all").lower()
        if sex_type != "all":
            adata = adata[adata.obs["sex"] == sex_type].copy()

        # Split-specific filtering for evaluation (only test groups)
        split_method = getattr(config.data.split, "method", "random").lower()
        
        if split_method == "genotype":
            test_genotype_spec = getattr(
                config.data.split.genotype, "test", "alzheimers"
            ).lower()
            
            if test_genotype_spec == "alzheimers":
                # Filter to only alzheimer's genotypes for evaluation
                test_genotypes = ["ab42", "htau"]
                adata = adata[
                    adata.obs["genotype"].str.lower().isin(test_genotypes)
                ].copy()
                print(f"Evaluation: Filtered to {adata.n_obs} cells with test genotypes {test_genotypes}")
            else:
                # Filter to specific test genotype
                adata = adata[
                    adata.obs["genotype"].str.lower() == test_genotype_spec
                ].copy()
                print(f"Evaluation: Filtered to {adata.n_obs} cells with genotype {test_genotype_spec}")
            
            # Apply test_ratio for evaluation performance if specified
            test_ratio = getattr(config.data.split.genotype, "test_ratio", 1.0)
            if test_ratio < 1.0:
                from sklearn.model_selection import train_test_split
                random_state = getattr(config.general, "random_state", 42)
                _, adata = train_test_split(
                    adata,
                    test_size=test_ratio,
                    random_state=random_state,
                )
                print(f"Evaluation: Applied test_ratio {test_ratio}, using {adata.n_obs} cells")
                
        elif split_method == "sex":
            # Filter to only test sex group for evaluation (matches training config structure)
            test_sex = getattr(
                config.data.train_test_split.test, "sex", "female"
            ).lower()
            adata = adata[adata.obs["sex"].str.lower() == test_sex].copy()
            print(f"Evaluation: Filtered to {adata.n_obs} cells with test sex '{test_sex}'")
            
        elif split_method == "tissue":
            # Filter to only test tissue group for evaluation (matches training config structure)
            test_tissue = getattr(
                config.data.train_test_split.test, "tissue", "body"
            ).lower()
            adata = adata[adata.obs["tissue"].str.lower() == test_tissue].copy()
            print(f"Evaluation: Filtered to {adata.n_obs} cells with test tissue '{test_tissue}'")

        # Apply gene selection from corrected data if specified and not already using corrected data
        select_batch_genes = getattr(
            config.preprocessing.genes, "select_batch_genes", False
        )

        if select_batch_genes and not batch_correction_enabled:
            if hasattr(self, "adata") and hasattr(self, "adata_corrected"):
                common_genes = self.adata.var_names.intersection(
                    self.adata_corrected.var_names
                )
                adata = adata[:, common_genes]

        # Highly variable genes selection
        if highly_variable_genes is not None:
            adata = adata[:, highly_variable_genes]
        else:
            # Use num_features to select the top genes
            adata = adata[:, adata.var_names[:num_features]]
            adata = adata[:, :num_features]

        # Prepare the testing labels
        encoding_var = getattr(config.data, "encoding_variable", "age")
        test_labels = label_encoder.transform(adata.obs[encoding_var])
        test_labels = to_categorical(test_labels)

        test_data = adata.X
        if issparse(test_data):  # Convert sparse matrices to dense
            test_data = test_data.toarray()

        if is_scaler_fit and scaler is not None:
            if not batch_correction_enabled:
                test_data = np.log1p(test_data)
            test_data = scaler.transform(test_data)

        # Reshape the testing data for CNN
        model_type = getattr(config.data, "model", "mlp").lower()
        if model_type == "cnn":
            test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

        return test_data, test_labels, label_encoder
