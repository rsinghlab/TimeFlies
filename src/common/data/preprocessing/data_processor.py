"""Main data preprocessing utilities for TimeFlies project."""

import logging
from typing import Any

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical  # type: ignore

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

        # Filter based on 'sex' if specified
        sex_type = getattr(config.data, "sex", "all").lower()
        if sex_type in ["male", "female"]:
            adata = adata[adata.obs["sex"] == sex_type].copy()

        # Filter based on 'cell_type' if specified
        # Check new format first (config.data.cell)
        cell_config = getattr(config.data, "cell", None)
        if cell_config:
            cell_type = getattr(cell_config, "type", "all").lower()
            if cell_type != "all":
                cell_type_column = getattr(cell_config, "column", None)
                if not cell_type_column:
                    raise ValueError("Cell type filtering enabled but no column specified in config.data.cell.column")
                # Validate cell type column exists (case-insensitive)
                actual_columns = adata.obs.columns.tolist()
                column_mapping = {col.lower(): col for col in actual_columns}
                cell_type_column_lower = cell_type_column.lower()
                
                if cell_type_column_lower not in column_mapping:
                    raise ValueError(f"Cell type column '{cell_type_column}' not found in data. Available columns: {actual_columns}")
                
                # Use the actual column name from the data
                actual_cell_column = column_mapping[cell_type_column_lower]
                available_types = adata.obs[actual_cell_column].str.lower().unique()
                if cell_type not in available_types:
                    available_types_orig = adata.obs[actual_cell_column].unique()
                    raise ValueError(f"Cell type '{cell_type}' not found in column '{actual_cell_column}'. Available types: {list(available_types_orig)}")
                adata = adata[adata.obs[actual_cell_column].str.lower() == cell_type].copy()
        else:
            # Check old format (config.data.cell_filtering)
            cell_filtering = getattr(config.data, "cell_filtering", None)
            if cell_filtering:
                cell_type = getattr(cell_filtering, "type", "all")
                if cell_type != "all":
                    cell_type_column = getattr(
                        cell_filtering, "column", "afca_annotation_broad"
                    )
                    # Validate cell type exists
                    if cell_type_column not in adata.obs.columns:
                        raise ValueError(f"Cell type column '{cell_type_column}' not found in data")
                    available_types = adata.obs[cell_type_column].unique()
                    if cell_type not in available_types:
                        raise ValueError(f"Cell type '{cell_type}' not found in column '{cell_type_column}'. Available types: {list(available_types)}")
                    adata = adata[adata.obs[cell_type_column] == cell_type].copy()
            else:
                # Fallback for backwards compatibility
                cell_type = getattr(config.data, "cell_type", "all")
                if cell_type != "all":
                    cell_type_column = getattr(
                        config.data, "cell_type_column", "afca_annotation_broad"
                    )
                    # Validate cell type exists
                    if cell_type_column not in adata.obs.columns:
                        raise ValueError(f"Cell type column '{cell_type_column}' not found in data")
                    available_types = adata.obs[cell_type_column].unique()
                    if cell_type not in available_types:
                        raise ValueError(f"Cell type '{cell_type}' not found in column '{cell_type_column}'. Available types: {list(available_types)}")
                    adata = adata[adata.obs[cell_type_column] == cell_type].copy()

        # Shuffle genes if required
        shuffle_genes = getattr(config.preprocessing.shuffling, "shuffle_genes", False)
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

    def split_data(
        self, dataset: AnnData, evaluation_mode: bool = False
    ) -> tuple[AnnData, AnnData]:
        """
        Filter/split the dataset based on the configuration using generic column-based approach.

        For training: Returns filtered training data + empty test set
        For evaluation: Returns empty train set + filtered test data

        Parameters:
        - dataset: The dataset to be split.
        - evaluation_mode: If True, return test data for evaluation

        Returns:
        - train_subset: The training subset (empty if evaluation_mode=True).
        - test_subset: The testing subset (empty if evaluation_mode=False).
        """
        config = self.config
        split_method = getattr(config.data.split, "method", "random").lower()

        if split_method == "column":
            # Generic column-based splitting
            column = getattr(config.data.split, "column", None)
            train_values = getattr(config.data.split, "train", [])
            test_values = getattr(config.data.split, "test", [])

            if not column:
                raise ValueError(
                    "Column name must be specified for column-based splits"
                )

            if column not in dataset.obs.columns:
                raise ValueError(f"Column '{column}' not found in dataset observations")

            # Convert values to lowercase strings for consistent matching
            train_values_norm = [str(v).lower() for v in train_values]
            test_values_norm = [str(v).lower() for v in test_values]

            # Filter data based on column values
            if evaluation_mode:
                # For evaluation: return test data only
                train_subset = dataset[:0].copy()  # Empty subset
                test_subset = dataset[
                    dataset.obs[column].str.lower().isin(test_values_norm)
                ].copy()
                print(
                    f"Evaluation mode: Using {test_subset.n_obs} cells with {column} in {test_values}"
                )
            else:
                # For training: return training data only
                train_subset = dataset[
                    dataset.obs[column].str.lower().isin(train_values_norm)
                ].copy()
                test_subset = dataset[:0].copy()  # Empty subset

            # Apply sampling (only to training data during training)
            num_samples = getattr(config.data.sampling, "samples", None)
            if num_samples and not evaluation_mode and num_samples < train_subset.n_obs:
                random_state = getattr(config.general, "random_state", 42)
                train_sample_indices = train_subset.obs.sample(
                    n=min(num_samples, train_subset.n_obs), random_state=random_state
                ).index
                train_subset = train_subset[train_sample_indices, :]
                # Sampling applied - details will be shown in training data logs

        else:
            # Random split method - no test_ratio, pass entire dataset
            if evaluation_mode:
                # For evaluation: return empty train, full dataset as test
                train_subset = dataset[:0].copy()  # Empty subset
                test_subset = dataset.copy()
                print(
                    f"Evaluation mode: Using full dataset ({test_subset.n_obs} cells)"
                )
            else:
                # For training: return full dataset for training
                # Keras will handle train/validation split internally
                train_subset = dataset.copy()
                test_subset = dataset[:0].copy()  # Empty subset
                # Training data will be shown in detailed preprocessing logs

            # Apply sampling if specified (only during training)
            num_samples = getattr(config.data.sampling, "samples", None)
            if num_samples and not evaluation_mode and num_samples < train_subset.n_obs:
                random_state = getattr(config.general, "random_state", 42)
                train_sample_indices = train_subset.obs.sample(
                    n=min(num_samples, train_subset.n_obs), random_state=random_state
                ).index
                train_subset = train_subset[train_sample_indices, :]
                # Sampling applied - details will be shown in training data logs

        return train_subset, test_subset

    def select_highly_variable_genes(
        self, train_subset: AnnData, test_subset: AnnData
    ) -> tuple[AnnData, AnnData, list[str] | None]:
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

        if getattr(config.preprocessing.genes, "highly_variable_genes", False):
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
    ) -> tuple[np.ndarray, np.ndarray, LabelEncoder | None]:
        """
        Prepare labels for training and testing data.

        Parameters:
        - train_subset: The training subset.
        - test_subset: The testing subset.

        Returns:
        - train_labels: Processed training labels (one-hot for classification, raw for regression).
        - test_labels: Processed testing labels.
        - label_encoder: The label encoder (classification only) or None (regression).
        """
        config = self.config
        encoding_var = getattr(config.data, "target_variable", "age")
        task_type = getattr(config.model, "task_type", "classification")

        if task_type == "regression":
            # For regression, keep labels as continuous values
            train_labels = train_subset.obs[encoding_var].values.astype(np.float32)
            if test_subset.n_obs > 0:
                test_labels = test_subset.obs[encoding_var].values.astype(np.float32)
            else:
                test_labels = np.array([], dtype=np.float32)

            # Reshape to (n_samples, 1) for regression
            train_labels = train_labels.reshape(-1, 1)
            test_labels = (
                test_labels.reshape(-1, 1)
                if len(test_labels) > 0
                else test_labels.reshape(0, 1)
            )

            return train_labels, test_labels, None
        else:
            # For classification, use label encoding and one-hot encoding
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(train_subset.obs[encoding_var])

            # Handle empty test set (simplified training approach)
            if test_subset.n_obs > 0:
                test_labels = label_encoder.transform(test_subset.obs[encoding_var])
                test_labels = to_categorical(test_labels)
            else:
                # Create empty test labels with correct shape
                test_labels = np.array([]).reshape(0, len(label_encoder.classes_))

            train_labels = to_categorical(train_labels)

            return train_labels, test_labels, label_encoder

    def normalize_data(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler | None, bool]:
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
    ) -> tuple[np.ndarray, np.ndarray]:
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
            # Using uncorrected data for training
            dataset_to_use = self.process_adata(self.adata)

        # Split the data
        train_subset, test_subset = self.split_data(dataset_to_use)

        # Select highly variable genes if required
        (
            train_subset,
            test_subset,
            highly_variable_genes,
        ) = self.select_highly_variable_genes(train_subset, test_subset)

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

        # Store the gene names used in training for consistent evaluation
        if highly_variable_genes is not None:
            self.train_gene_names = highly_variable_genes
        else:
            self.train_gene_names = train_subset.var_names.tolist()

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
            train_subset,  # Add metadata-rich subset for display
            test_subset,  # Add metadata-rich subset for display
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
        train_gene_names=None,
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
            filtered_adata (AnnData): Filtered AnnData object for display and metrics (with original metadata)
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
        # Check new format first (config.data.cell)
        cell_config = getattr(config.data, "cell", None)
        if cell_config:
            cell_type = getattr(cell_config, "type", "all").lower()
            if cell_type != "all":
                cell_type_column = getattr(cell_config, "column", None)
                if not cell_type_column:
                    raise ValueError("Cell type filtering enabled but no column specified in config.data.cell.column")
                # Validate cell type column exists (case-insensitive)
                actual_columns = adata.obs.columns.tolist()
                column_mapping = {col.lower(): col for col in actual_columns}
                cell_type_column_lower = cell_type_column.lower()
                
                if cell_type_column_lower not in column_mapping:
                    raise ValueError(f"Cell type column '{cell_type_column}' not found in data. Available columns: {actual_columns}")
                
                # Use the actual column name from the data
                actual_cell_column = column_mapping[cell_type_column_lower]
                available_types = adata.obs[actual_cell_column].str.lower().unique()
                if cell_type not in available_types:
                    available_types_orig = adata.obs[actual_cell_column].unique()
                    raise ValueError(f"Cell type '{cell_type}' not found in column '{actual_cell_column}'. Available types: {list(available_types_orig)}")
                adata = adata[adata.obs[actual_cell_column].str.lower() == cell_type].copy()
        else:
            # Check old format (config.data.cell_filtering)
            cell_filtering = getattr(config.data, "cell_filtering", None)
            if cell_filtering:
                cell_type = getattr(cell_filtering, "type", "all").lower()
                if cell_type != "all":
                    cell_type_column = getattr(
                        cell_filtering, "column", "afca_annotation_broad"
                    )
                    # Validate cell type exists
                    if cell_type_column not in adata.obs.columns:
                        raise ValueError(f"Cell type column '{cell_type_column}' not found in data")
                    available_types = adata.obs[cell_type_column].unique()
                    if cell_type not in available_types:
                        raise ValueError(f"Cell type '{cell_type}' not found in column '{cell_type_column}'. Available types: {list(available_types)}")
                    adata = adata[adata.obs[cell_type_column] == cell_type].copy()
            else:
                # Fallback for backwards compatibility
                cell_type = getattr(config.data, "cell_type", "all").lower()
                if cell_type != "all":
                    cell_type_column = getattr(
                        config.data, "cell_type_column", "afca_annotation_broad"
                    )
                    # Validate cell type exists
                    if cell_type_column not in adata.obs.columns:
                        raise ValueError(f"Cell type column '{cell_type_column}' not found in data")
                    available_types = adata.obs[cell_type_column].unique()
                    if cell_type not in available_types:
                        raise ValueError(f"Cell type '{cell_type}' not found in column '{cell_type_column}'. Available types: {list(available_types)}")
                    adata = adata[adata.obs[cell_type_column] == cell_type].copy()

        # Sex Mapping
        sex_type = getattr(config.data, "sex", "all").lower()
        if sex_type != "all":
            adata = adata[adata.obs["sex"] == sex_type].copy()

        # Split-specific filtering for evaluation (only test groups)
        split_method = getattr(config.data.split, "method", "random").lower()

        if split_method == "column":
            # Generic column-based evaluation filtering
            column = getattr(config.data.split, "column", None)
            test_values = getattr(config.data.split, "test", [])

            if column and test_values:
                # Convert values to lowercase strings for consistent matching
                test_values_norm = [str(v).lower() for v in test_values]

                # Filter to only test values for evaluation
                adata = adata[
                    adata.obs[column].str.lower().isin(test_values_norm)
                ].copy()
                # Evaluation filtering applied silently

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

        # Gene selection - use exact same genes as training
        if train_gene_names is not None:
            # Use the exact gene names from training for consistency
            adata = adata[:, train_gene_names]
        elif highly_variable_genes is not None:
            adata = adata[:, highly_variable_genes]
        else:
            # Use num_features to select the top genes
            adata = adata[:, adata.var_names[:num_features]]
            adata = adata[:, :num_features]

        # Prepare the testing labels based on task type
        encoding_var = getattr(config.data, "target_variable", "age")
        task_type = getattr(config.model, "task_type", "classification")

        if task_type == "regression":
            # For regression, keep as continuous values
            test_labels = (
                adata.obs[encoding_var].values.astype(np.float32).reshape(-1, 1)
            )
        else:
            # For classification, use label encoding and one-hot encoding
            # Handle case where evaluation data might have fewer classes than training
            eval_classes = adata.obs[encoding_var].unique()
            missing_classes = set(label_encoder.classes_) - set(eval_classes)

            if missing_classes:
                print(f"Warning: Evaluation data missing classes: {missing_classes}")
                print(
                    f"Model will predict across full {len(label_encoder.classes_)} class space"
                )

            test_labels = label_encoder.transform(adata.obs[encoding_var])
            test_labels = to_categorical(
                test_labels, num_classes=len(label_encoder.classes_)
            )

        test_data = adata.X
        if issparse(test_data):  # Convert sparse matrices to dense
            test_data = test_data.toarray()

        if is_scaler_fit and scaler is not None:
            if not batch_correction_enabled:
                test_data = np.log1p(test_data)
            test_data = scaler.transform(test_data)

        # Keep a copy of the filtered AnnData for display/metrics (before reshaping)
        filtered_adata = adata.copy()

        # Reshape the testing data for CNN
        model_type = getattr(config.data, "model", "mlp").lower()
        if model_type == "cnn":
            test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

        return test_data, test_labels, label_encoder, filtered_adata
