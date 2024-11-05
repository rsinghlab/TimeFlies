# preprocessing.py

import os
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import dill as pickle
import logging
from utilities import (
    PathManager,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle data preprocessing.

    This class preprocesses the input data based on the configuration settings or loads
    preprocessed data if no further processing is required.
    """

    def __init__(self, config, adata, adata_corrected):
        """
        Initializes the DataPreprocessor with the given configuration and datasets.

        Parameters:
        - config (Config): The configuration object containing all necessary parameters.
        - adata (AnnData): The main AnnData object containing the dataset.
        - adata_corrected (AnnData): The batch-corrected AnnData object.
        """
        self.config = config
        self.adata = adata
        self.adata_corrected = adata_corrected

        # Initialize PathManager
        self.path_manager = PathManager(config)

    def process_adata(self, adata):
        """
        Process an AnnData object based on the provided configuration.

        Parameters:
        - adata (AnnData): The AnnData object to be processed.

        Returns:
        - adata (AnnData): The processed AnnData object.
        """
        config = self.config

        # Include or exclude 'mix' sex
        include_mix = config.DataParameters.Filtering.include_mixed_sex
        if not include_mix:
            adata = adata[adata.obs["sex"] != "mix"].copy()

        # Filter based on 'sex_type' if specified
        sex_type = config.DataParameters.GeneralSettings.sex_type.lower()
        if sex_type in ["male", "female"]:
            adata = adata[adata.obs["sex"] == sex_type].copy()

        # Filter based on 'cell_type' if specified
        cell_type = config.DataParameters.GeneralSettings.cell_type
        if cell_type != "all":
            adata = adata[adata.obs["afca_annotation_broad"] == cell_type].copy()

        # Shuffle genes if required
        shuffle_genes = config.GenePreprocessing.GeneShuffle.shuffle_genes
        shuffle_random_state = config.GenePreprocessing.GeneShuffle.shuffle_random_state
        if shuffle_genes:
            # Create a Generator with a fixed seed
            rng = np.random.default_rng(
                shuffle_random_state
            )  # You can choose any seed value here
            gene_order = rng.permutation(adata.var_names)
            adata = adata[:, gene_order]

        # Sample data if required
        num_samples = config.DataParameters.Sampling.num_samples
        if num_samples and num_samples < adata.n_obs:
            sample_indices = adata.obs.sample(
                n=num_samples, random_state=config.DataSplit.random_state
            ).index
            adata = adata[sample_indices, :]

        # Select variables (genes) if required
        num_variables = config.DataParameters.Sampling.num_variables
        if num_variables and num_variables < adata.n_vars:
            adata = adata[:, adata.var_names[:num_variables]]

        return adata

    def split_data(self, dataset):
        """
        Split the dataset into training and testing subsets based on the configuration.

        Parameters:
        - dataset (AnnData): The dataset to be split.

        Returns:
        - train_subset (AnnData): The training subset.
        - test_subset (AnnData): The testing subset.
        """
        config = self.config
        split_method = config.DataParameters.TrainTestSplit.method.lower()

        if split_method == "sex":
            train_sex = config.DataParameters.TrainTestSplit.train.sex.lower()
            test_sex = config.DataParameters.TrainTestSplit.test.sex.lower()

            train_subset = dataset[dataset.obs["sex"].str.lower() == train_sex].copy()
            test_subset = dataset[dataset.obs["sex"].str.lower() == test_sex].copy()

            _, test_subset = train_test_split(
                test_subset,
                test_size=config.DataParameters.TrainTestSplit.test.size,
                random_state=config.DataSplit.random_state,
            )

        elif split_method == "tissue":
            train_tissue = config.DataParameters.TrainTestSplit.train.tissue.lower()
            test_tissue = config.DataParameters.TrainTestSplit.test.tissue.lower()

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

            _, test_subset = train_test_split(
                test_subset,
                test_size=config.DataParameters.TrainTestSplit.test.size,
                random_state=config.DataSplit.random_state,
            )

        else:
            # Perform a stratified train-test split based on encoding variable
            encoding_var = config.DataParameters.GeneralSettings.encoding_variable
            test_size = config.DataSplit.test_split
            random_state = config.DataSplit.random_state

            train_subset, test_subset = train_test_split(
                dataset,
                stratify=dataset.obs[encoding_var],
                test_size=test_size,
                random_state=random_state,
            )

            # Apply gene selection from corrected data if specified and not already using corrected data
            select_batch_genes = (
                config.GenePreprocessing.GeneFiltering.select_batch_genes
            )
            batch_correction_enabled = config.DataParameters.BatchCorrection.enabled
            if select_batch_genes and not batch_correction_enabled:
                adata_corrected_processed = self.process_adata(self.adata_corrected)
                common_genes = train_subset.var_names.intersection(
                    adata_corrected_processed.var_names
                )
                train_subset = train_subset[:, common_genes]
                test_subset = test_subset[:, common_genes]

        return train_subset, test_subset

    def select_highly_variable_genes(self, train_subset, test_subset):
        """
        Select highly variable genes from the training data and apply to both training and testing data.

        Parameters:
        - train_subset (AnnData): The training subset.
        - test_subset (AnnData): The testing subset.

        Returns:
        - train_subset (AnnData): The training subset with selected genes.
        - test_subset (AnnData): The testing subset with selected genes.
        - highly_variable_genes (list): List of highly variable genes.
        """
        config = self.config
        highly_variable_genes = None

        if config.GenePreprocessing.GeneFiltering.highly_variable_genes:
            normal = train_subset.copy()
            batch_correction_enabled = config.DataParameters.BatchCorrection.enabled
            if not batch_correction_enabled:
                sc.pp.normalize_total(normal, target_sum=1e4)
                sc.pp.log1p(normal)
            sc.pp.highly_variable_genes(normal, n_top_genes=5000)
            highly_variable_genes = normal.var_names[
                normal.var.highly_variable
            ].tolist()
            train_subset = train_subset[:, highly_variable_genes]
            test_subset = test_subset[:, highly_variable_genes]

        return train_subset, test_subset, highly_variable_genes

    def prepare_labels(self, train_subset, test_subset):
        """
        Prepare labels for training and testing data.

        Parameters:
        - train_subset (AnnData): The training subset.
        - test_subset (AnnData): The testing subset.

        Returns:
        - train_labels (ndarray): One-hot encoded training labels.
        - test_labels (ndarray): One-hot encoded testing labels.
        - label_encoder (LabelEncoder): The label encoder used to transform the labels.
        """
        config = self.config
        encoding_var = config.DataParameters.GeneralSettings.encoding_variable
        label_encoder = LabelEncoder()

        train_labels = label_encoder.fit_transform(train_subset.obs[encoding_var])
        test_labels = label_encoder.transform(test_subset.obs[encoding_var])

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        return train_labels, test_labels, label_encoder

    def normalize_data(self, train_data, test_data):
        """
        Normalize training and testing data if required.

        Parameters:
        - train_data (ndarray): Training data.
        - test_data (ndarray): Testing data.

        Returns:
        - train_data (ndarray): Normalized training data.
        - test_data (ndarray): Normalized testing data.
        - scaler (StandardScaler or None): The scaler used for normalization.
        - is_scaler_fit (bool): Flag indicating whether the scaler was fit (i.e., normalization was applied).
        """
        config = self.config
        normalization_enabled = config.DataProcessing.Normalization.enabled

        if normalization_enabled:
            logging.info("Normalizing data...")
            train_data = np.log1p(train_data)
            test_data = np.log1p(test_data)

            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            is_scaler_fit = True
            logging.info("Data normalized.")
        else:
            scaler = None  # No scaling applied
            is_scaler_fit = False

        return train_data, test_data, scaler, is_scaler_fit

    def reshape_for_cnn(self, train_data, test_data):
        """
        Reshape data for CNN models.

        Parameters:
        - train_data (ndarray): Training data.
        - test_data (ndarray): Testing data.

        Returns:
        - train_data (ndarray): Reshaped training data.
        - test_data (ndarray): Reshaped testing data.
        """
        train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
        test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
        return train_data, test_data

    def create_reference_data(self, train_data):
        """
        Create reference data by sampling from training data.

        Parameters:
        - train_data (ndarray): Training data.

        Returns:
        - reference_data (ndarray): Reference data.
        """
        config = self.config
        reference_size = config.FeatureImportanceAndVisualizations.reference_size
        if reference_size > train_data.shape[0]:
            reference_size = train_data.shape[0]

        reference_data = train_data[
            np.random.choice(train_data.shape[0], reference_size, replace=False)
        ]
        return reference_data

    def save_processed_data(
        self,
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
    ):
        """
        Save the processed data to disk.

        Parameters:
        - train_data (ndarray): Training data.
        - test_data (ndarray): Testing data.
        - train_labels (ndarray): Training labels.
        - test_labels (ndarray): Testing labels.
        - label_encoder (LabelEncoder): The label encoder.
        - reference_data (ndarray): Reference data.
        - scaler (StandardScaler or None): The scaler used for normalization.
        - highly_variable_genes (list): List of highly variable genes.
        - mix_included (bool): Flag indicating if 'mix' sex is included.
        """
        processed_data_dir = self.path_manager.get_processed_data_dir()
        os.makedirs(processed_data_dir, exist_ok=True)

        # Save the preprocessed data
        np.save(
            os.path.join(processed_data_dir, "processed_train_data.npy"), train_data
        )
        np.save(
            os.path.join(processed_data_dir, "processed_train_labels.npy"), train_labels
        )
        np.save(os.path.join(processed_data_dir, "processed_test_data.npy"), test_data)
        np.save(
            os.path.join(processed_data_dir, "processed_test_labels.npy"), test_labels
        )
        np.save(os.path.join(processed_data_dir, "reference_data.npy"), reference_data)

        # Save the LabelEncoder
        with open(os.path.join(processed_data_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

        # Save the Scaler if used
        if scaler is not None:
            with open(os.path.join(processed_data_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)

        # Save the is_scaler_fit flag
        with open(os.path.join(processed_data_dir, "is_scaler_fit.pkl"), "wb") as f:
            pickle.dump(is_scaler_fit, f)

        # Save the highly variable genes
        with open(
            os.path.join(processed_data_dir, "highly_variable_genes.pkl"), "wb"
        ) as f:
            pickle.dump(highly_variable_genes, f)

        # Save the mix_included flag
        with open(os.path.join(processed_data_dir, "mix_included.pkl"), "wb") as f:
            pickle.dump(mix_included, f)

    def load_processed_data(self):
        """
        Loads the preprocessed data from the specified directory.

        Returns:
            train_data (ndarray): Training data.
            test_data (ndarray): Testing data.
            train_labels (ndarray): Labels for the training data.
            test_labels (ndarray): Labels for the testing data.
            label_encoder (LabelEncoder): The label encoder used to transform the labels.
            reference_data (ndarray): Reference data.
            scaler (StandardScaler or None): The scaler used for data normalization.
            highly_variable_genes (list): List of highly variable genes.
            mix_included (bool): Flag indicating if 'mix' sex is included.
        """
        processed_data_dir = self.path_manager.get_processed_data_dir()
        logger.info(f"Loading data from {processed_data_dir}")

        required_files = [
            "processed_train_data.npy",
            "processed_train_labels.npy",
            "processed_test_data.npy",
            "processed_test_labels.npy",
            "reference_data.npy",
            "label_encoder.pkl",
            "highly_variable_genes.pkl",
            "mix_included.pkl",
            "scaler.pkl",
            "is_scaler_fit.pkl",
        ]

        # Check if all required files are present
        for filename in required_files:
            if not os.path.isfile(os.path.join(processed_data_dir, filename)):
                raise FileNotFoundError(
                    f"Required file '{filename}' not found in directory '{processed_data_dir}'"
                )

        # Load the preprocessed data
        train_data = np.load(
            os.path.join(processed_data_dir, "processed_train_data.npy")
        )
        train_labels = np.load(
            os.path.join(processed_data_dir, "processed_train_labels.npy")
        )
        test_data = np.load(os.path.join(processed_data_dir, "processed_test_data.npy"))
        test_labels = np.load(
            os.path.join(processed_data_dir, "processed_test_labels.npy")
        )
        reference_data = np.load(os.path.join(processed_data_dir, "reference_data.npy"))

        with open(os.path.join(processed_data_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        with open(
            os.path.join(processed_data_dir, "highly_variable_genes.pkl"), "rb"
        ) as f:
            highly_variable_genes = pickle.load(f)
        with open(os.path.join(processed_data_dir, "mix_included.pkl"), "rb") as f:
            mix_included = pickle.load(f)

        # Load the is_scaler_fit flag
        with open(os.path.join(processed_data_dir, "is_scaler_fit.pkl"), "rb") as f:
            is_scaler_fit = pickle.load(f)

        scaler_path = os.path.join(processed_data_dir, "scaler.pkl")
        if os.path.isfile(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = None

        # Print the shape of the training and test datasets
        print("Shape of Training Data:", train_data.shape)
        print("Shape of Test Data:", test_data.shape)
        print("Shape of Training Labels:", train_labels.shape)
        print("Shape of Test Labels:", test_labels.shape)

        # Return the loaded data including is_scaler_fit
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

    def preprocess_and_prepare_data(self):
        """
        Preprocess the data based on the provided configuration parameters.

        Returns:
            train_data (ndarray): Training data.
            test_data (ndarray): Testing data.
            train_labels (ndarray): Labels for the training data.
            test_labels (ndarray): Labels for the testing data.
            label_encoder (LabelEncoder): The label encoder used to transform the labels.
            reference_data (ndarray): Reference data.
            scaler (StandardScaler or None): The scaler used for data normalization.
            highly_variable_genes (list): List of highly variable genes.
            mix_included (bool): Flag indicating if 'mix' sex is included.
        """
        config = self.config

        adata = self.process_adata(self.adata)
        adata_corrected = self.process_adata(self.adata_corrected)

        # Decide which dataset to use
        batch_correction_enabled = config.DataParameters.BatchCorrection.enabled
        dataset_to_use = adata_corrected if batch_correction_enabled else adata

        # Split the data
        train_subset, test_subset = self.split_data(dataset_to_use)

        # Select highly variable genes if required
        train_subset, test_subset, highly_variable_genes = (
            self.select_highly_variable_genes(train_subset, test_subset)
        )

        # Print data sizes and class counts
        print(f"Training data size: {train_subset.shape}")
        print(f"Testing data size: {test_subset.shape}")

        encoding_var = config.DataParameters.GeneralSettings.encoding_variable
        print("\nCounts of each class in the training data:")
        print(train_subset.obs[encoding_var].value_counts())

        print("\nCounts of each class in the testing data:")
        print(test_subset.obs[encoding_var].value_counts())

        # Prepare labels
        train_labels, test_labels, label_encoder = self.prepare_labels(
            train_subset, test_subset
        )

        # Prepare data arrays
        train_data = train_subset.X
        if issparse(train_data):
            train_data = train_data.toarray()

        test_data = test_subset.X
        if issparse(test_data):
            test_data = test_data.toarray()

        # Normalize data if required
        train_data, test_data, scaler, is_scaler_fit = self.normalize_data(
            train_data, test_data
        )

        # Reshape data for CNN if required
        model_type = config.DataParameters.GeneralSettings.model_type.lower()
        if model_type == "cnn":
            train_data, test_data = self.reshape_for_cnn(train_data, test_data)

        # Create reference data
        reference_data = self.create_reference_data(train_data)

        # Save processed data if required
        save_data = config.DataProcessing.Preprocessing.save_data
        if save_data:
            mix_included = config.DataParameters.Filtering.include_mixed_sex
            self.save_processed_data(
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

        Args:
            adata (AnnData): The input data.
            label_encoder (LabelEncoder): The label encoder used to transform the labels.
            num_features (int): The number of features to use.
            scaler (StandardScaler or None): The scaler used for data normalization.
            is_scaler_fit (bool): Flag indicating if the scaler was fit.
            highly_variable_genes (list): List of highly variable genes.
            mix_included (bool): Flag indicating if 'mix'

        Returns:
            test_data (ndarray): Testing data.
            test_labels (ndarray): Labels for the testing data.
            label_encoder (LabelEncoder): The label encoder used to transform the labels.
        """
        config = self.config

        # Remove 'mix' if specified
        if mix_included is False:
            adata = adata[adata.obs["sex"] != "mix"].copy()

        # Check if the specified cell type is 'all'
        cell_type = config.DataParameters.GeneralSettings.cell_type.lower()
        if cell_type != "all":
            adata = adata[adata.obs["afca_annotation_broad"] == cell_type].copy()

        # Sex Mapping
        sex_type = config.DataParameters.GeneralSettings.sex_type.lower()
        if sex_type != "all":
            adata = adata[adata.obs["sex"] == sex_type].copy()

        # Apply gene selection from corrected data if specified and not already using corrected data
        select_batch_genes = config.GenePreprocessing.GeneFiltering.select_batch_genes
        batch_correction_enabled = config.DataParameters.BatchCorrection.enabled

        if select_batch_genes and not batch_correction_enabled:
            common_genes = self.adata.var_names.intersection(self.adata_corrected.var_names)
            adata = adata[:, common_genes]
        
        # Highly variable genes selection
        if highly_variable_genes is not None:
            adata = adata[:, highly_variable_genes]
        else:
            # Use num_features to select the top genes
            adata = adata[:, adata.var_names[:num_features]]
            adata = adata[:, :num_features]

        # Prepare the testing labels
        encoding_var = config.DataParameters.GeneralSettings.encoding_variable
        test_labels = label_encoder.transform(adata.obs[encoding_var])
        test_labels = to_categorical(test_labels)

        test_data = adata.X
        if issparse(test_data):  # Convert sparse matrices to dense
            test_data = test_data.toarray()

        if is_scaler_fit and scaler is not None:
            test_data = np.log1p(test_data)
            test_data = scaler.transform(test_data)

        # Reshape the testing data for CNN
        model_type = config.DataParameters.GeneralSettings.model_type.lower()
        if model_type == "cnn":
            test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

        # # **Extract the feature names here**
        # relevant_feature_names = adata.var_names.tolist()

        return test_data, test_labels, label_encoder

    def prepare_data(self):
        """
        Preprocesses the data or loads preprocessed data based on the configuration.

        Returns:
        - tuple: A tuple containing preprocessed train data, test data, labels, and other related objects.
        """
        config = self.config

        # Load or preprocess data based on configuration
        preprocess_required = config.DataProcessing.Preprocessing.required
        if not preprocess_required:
            data = self.load_processed_data()
        else:
            data = self.preprocess_and_prepare_data()

        # Unpack train data
        train_data = data[0]

        # Determine the number of input variables based on model type
        model_type = config.DataParameters.GeneralSettings.model_type.lower()
        if model_type == "cnn":
            config.DataParameters.Sampling.num_variables = train_data.shape[2]
        else:
            config.DataParameters.Sampling.num_variables = train_data.shape[1]
        return data


class GeneFilter:
    """
    A class to handle gene filtering based on configuration settings.

    This class takes in multiple datasets and gene lists, and applies filters based on
    the configuration provided.
    """

    def __init__(
        self, config, adata, adata_eval, adata_original, autosomal_genes, sex_genes
    ):
        """
        Initializes the GeneFilter with the given configuration and datasets.

        Parameters:
        - config (Config): The configuration object containing all necessary parameters.
        - adata (AnnData): The main AnnData object containing the dataset.
        - adata_eval (AnnData): The AnnData object for evaluation data.
        - adata_original (AnnData): The original unfiltered AnnData object.
        - autosomal_genes (list): A list of autosomal genes.
        - sex_genes (list): A list of sex-linked genes.
        """
        self.config = config
        self.adata = adata
        self.adata_eval = adata_eval
        self.adata_original = adata_original
        self.autosomal_genes = autosomal_genes
        self.sex_genes = sex_genes

    def balance_genes(self, gene_type_1, gene_type_2):
        """
        Balance the number of genes in gene_type_1 to match the number of genes in gene_type_2.

        Parameters:
        gene_type_1 (list): List of genes of the first type (e.g., autosomal genes).
        gene_type_2 (list): List of genes of the second type (e.g., X genes).

        Returns:
        balanced_genes (list): Subset of gene_type_1 with the same size as the number of genes in gene_type_2.
        """
        config = self.config
        np.random.seed(config.DataSplit.random_state)
        num_genes_type_2 = len(gene_type_2)
        if num_genes_type_2 > len(gene_type_1):
            raise ValueError(
                "Number of genes in gene_type_2 is greater than the number of genes in gene_type_1 available."
            )

        # Select a random subset of gene_type_1 to match the size of gene_type_2
        balanced_genes = np.random.choice(
            gene_type_1, num_genes_type_2, replace=False
        ).tolist()
        return balanced_genes

    def create_and_apply_mask(
        self,
        data,
        sex_genes,
        autosomal_genes,
        original_autosomal_genes,
        lnc_genes=None,
    ):
        """
        Create and apply masks to filter genes in the dataset based on configuration.

        Parameters:
        data (AnnData): Annotated data matrix with genes as variables.
        sex_genes (list): List of sex-linked genes.
        autosomal_genes (list): List of autosomal genes after balancing (if applicable).
        original_autosomal_genes (list): Original list of autosomal genes before any balancing.
        lnc_genes (list, optional): List of long non-coding RNA genes, if applicable.

        Returns:
        data (AnnData): Filtered data matrix based on the applied masks.
        """
        config = self.config

        # Create masks for sex genes and autosomal genes
        sex_mask = data.var.index.isin(sex_genes)
        autosomal_mask = data.var.index.isin(autosomal_genes)
        original_autosomal_mask = data.var.index.isin(original_autosomal_genes)

        # Create lncRNA mask
        if lnc_genes is not None:
            lnc_mask = data.var.index.isin(lnc_genes)
        else:
            lnc_mask = data.var_names.str.startswith("lnc")
        no_lnc_mask = ~lnc_mask

        # Initialize final_mask as all True
        final_mask = np.ones(data.shape[1], dtype=bool)

        # Remove unaccounted genes based on the original set of autosomal and sex genes
        if config.GenePreprocessing.GeneFiltering.remove_unaccounted_genes:
            accounted_mask = original_autosomal_mask | sex_mask
            data = data[:, accounted_mask]
            # Recompute the masks since data has changed
            sex_mask = data.var.index.isin(sex_genes)
            autosomal_mask = data.var.index.isin(autosomal_genes)
            original_autosomal_mask = data.var.index.isin(original_autosomal_genes)
            if lnc_genes is not None:
                lnc_mask = data.var.index.isin(lnc_genes)
            else:
                lnc_mask = data.var_names.str.startswith("lnc")
            no_lnc_mask = ~lnc_mask
        else:
            # Include genes not found in the provided gene lists
            unaccounted_mask = ~(original_autosomal_mask | sex_mask)

        # Apply various filters based on configuration settings
        only_keep_lnc = config.GenePreprocessing.GeneFiltering.only_keep_lnc_genes
        if only_keep_lnc:
            final_mask &= lnc_mask

        remove_autosomal = config.GenePreprocessing.GeneFiltering.remove_autosomal_genes
        if remove_autosomal:
            final_mask &= ~autosomal_mask

        remove_sex = config.GenePreprocessing.GeneFiltering.remove_sex_genes
        if remove_sex:
            final_mask &= ~sex_mask

        remove_lnc = config.GenePreprocessing.GeneFiltering.remove_lnc_genes
        if remove_lnc:
            final_mask &= no_lnc_mask

        # If not removing unaccounted genes, ensure they are included in the final mask
        if not config.GenePreprocessing.GeneFiltering.remove_unaccounted_genes:
            final_mask |= unaccounted_mask

        # Apply the final combined mask
        data = data[:, final_mask]

        return data

    def filter_genes_based_on_config(
        self, adata, adata_eval, adata_original, sex_genes, autosomal_genes
    ):
        """
        Filter genes in multiple datasets based on provided configurations.

        Parameters:
        adata (AnnData): Annotated data matrix for primary analysis.
        adata_eval (AnnData): Annotated data matrix for evaluation purposes.
        adata_original (AnnData): Original annotated data matrix for reference.
        sex_genes (list): List of sex-linked genes.
        autosomal_genes (list): List of autosomal genes.

        Returns:
        Tuple[AnnData, AnnData, AnnData]: Filtered versions of adata, adata_eval, and adata_original.
        """
        config = self.config

        # Derive lncRNA genes and non-lncRNA genes
        lnc_genes = adata.var_names[adata.var_names.str.startswith("lnc")].tolist()
        non_lnc_genes = [gene for gene in adata.var.index if not gene.startswith("lnc")]

        # Store the original autosomal genes before any balancing
        original_autosomal_genes = autosomal_genes.copy()

        # Balance the number of autosomal genes with the number of X genes if required
        balance_genes = config.GenePreprocessing.GeneBalancing.balance_genes
        if balance_genes:
            autosomal_genes = self.balance_genes(autosomal_genes, sex_genes)

        # Balance the number of non-lnc genes with the number of lnc genes if required
        balance_lnc_genes = config.GenePreprocessing.GeneBalancing.balance_lnc_genes
        if balance_lnc_genes:
            lnc_genes = self.balance_genes(non_lnc_genes, lnc_genes)

        # Apply the balanced genes masks to the datasets
        adata = self.create_and_apply_mask(
            data=adata,
            sex_genes=sex_genes,
            autosomal_genes=autosomal_genes,
            original_autosomal_genes=original_autosomal_genes,
            lnc_genes=lnc_genes if balance_lnc_genes else None,
        )
        adata_eval = self.create_and_apply_mask(
            data=adata_eval,
            sex_genes=sex_genes,
            autosomal_genes=autosomal_genes,
            original_autosomal_genes=original_autosomal_genes,
            lnc_genes=lnc_genes if balance_lnc_genes else None,
        )
        adata_original = self.create_and_apply_mask(
            data=adata_original,
            sex_genes=sex_genes,
            autosomal_genes=autosomal_genes,
            original_autosomal_genes=original_autosomal_genes,
            lnc_genes=lnc_genes if balance_lnc_genes else None,
        )

        return adata, adata_eval, adata_original

    def apply_filter(self):
        """
        Applies gene filters based on the configuration settings.

        This method filters the genes in the datasets according to the configuration
        settings, potentially removing autosomal genes, sex-linked genes, or other
        specified gene groups.

        Returns:
        - tuple: A tuple containing the filtered AnnData objects for the main, evaluation,
                 and original datasets.
        """
        # Apply the gene filtering based on the config
        self.adata, self.adata_eval, self.adata_original = (
            self.filter_genes_based_on_config(
                self.adata,
                self.adata_eval,
                self.adata_original,
                self.sex_genes,
                self.autosomal_genes,
            )
        )
        return self.adata, self.adata_eval, self.adata_original
