import tensorflow as tf
import os
import scanpy as sc
import pandas as pd


class GPUHandler:
    """
    A class to handle GPU configuration for TensorFlow.

    This class contains a static method to check the availability of GPUs
    and configure TensorFlow to allow memory growth on available GPUs.
    """

    @staticmethod
    def configure(config):
        """
        Configures TensorFlow to use available GPUs.

        Parameters:
        - config (Config): The configuration object.

        This method checks if any GPUs are available for TensorFlow. If available,
        it sets memory growth on each GPU to prevent TensorFlow from allocating all the
        GPU memory at once, allowing other processes to use the memory as needed.

        If the processor is 'M' (Mac M1/M2/M3), it configures TensorFlow accordingly.

        If memory growth cannot be set (e.g., because TensorFlow has already been
        initialized), it catches and prints the RuntimeError.
        """
        processor = config.Device.processor

        if processor == "M":
            # Configure TensorFlow for Apple M1/M2/M3 processors
            try:
                # Set up TensorFlow to use the 'metal' device (for Apple Silicon)
                tf.config.set_visible_devices([], "GPU")
                tf.config.set_visible_devices([], "XLA_GPU")
                tf.config.set_visible_devices([], "XLA_CPU")
                print("Configured TensorFlow for Apple Silicon processors.")
            except Exception as e:
                print("Could not set visible devices for M processor:", e)
        else:
            # Regular GPU configuration
            # Check if any GPUs are available for TensorFlow
            tf_gpu = len(tf.config.list_physical_devices("GPU")) > 0

            # Get the list of physical GPU devices recognized by TensorFlow
            gpus = tf.config.experimental.list_physical_devices("GPU")

            if gpus:
                try:
                    # Iterate over each available GPU and set memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth set successfully.")
                except RuntimeError as e:
                    # Catch and print exception if memory growth setting fails
                    print("Error setting GPU memory growth:", e)
            else:
                print("No GPUs found. Running on CPU.")


class DataLoader:
    """
    A class to handle data loading operations.

    This class manages the loading of datasets and gene lists from specified file paths
    based on the configuration provided.
    """

    def __init__(self, config):
        """
        Initializes the DataLoader with the given configuration.

        Parameters:
        - config (Config): The configuration object.
        """
        self.config = config

        # Get the directory of the current script
        self.Code_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the data directory based on the configuration
        self.Data_dir = os.path.join(
            self.Code_dir,
            "..",
            "Data",
            "h5ad",
        )

        # Prepare the paths to various data files
        self._prepare_paths()

    def _prepare_paths(self):
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

    def load_data(self):
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

    def load_corrected_data(self):
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

    def load_gene_lists(self):
        """
        Loads gene lists from CSV files.

        This method loads lists of autosomal and sex-linked genes from CSV files.

        Returns:
        - tuple: A tuple containing two lists, one for autosomal genes and one for sex-linked genes.
        """
        # Load autosomal gene list from a CSV file
        autosomal_genes = (
            pd.read_csv(
                os.path.join(self.Data_dir, "..", "autosomal.csv"),
                header=None,
                dtype=str,
            )
            .iloc[:, 0]
            .tolist()
        )

        # Load sex-linked gene list from a CSV file
        sex_genes = (
            pd.read_csv(os.path.join(self.Data_dir, "..", "sex.csv"), header=None)
            .iloc[:, 0]
            .tolist()
        )

        return autosomal_genes, sex_genes


class PathManager:
    """
    A utility class to construct model and visualization directory paths based on the configuration.

    This class centralizes the directory path construction logic to avoid code duplication
    between different components of the system.
    """

    def __init__(self, config):
        """
        Initializes the PathManager with the given configuration.

        Parameters:
        - config (Config): The configuration object containing all necessary parameters.
        """
        self.config = config

        # Precompute commonly used parameters
        self.batch_correction_enabled = (
            self.config.DataParameters.BatchCorrection.enabled
        )
        self.correction_dir = (
            "batch_corrected" if self.batch_correction_enabled else "uncorrected"
        )

        self.tissue = self.config.DataParameters.GeneralSettings.tissue.lower()

        self.model_type = self.config.DataParameters.GeneralSettings.model_type.upper()
        self.encoding_variable = (
            self.config.DataParameters.GeneralSettings.encoding_variable.lower()
        )
        self.cell_type = self.config.DataParameters.GeneralSettings.cell_type.lower()
        self.sex_type = self.config.DataParameters.GeneralSettings.sex_type.lower()

        self.num_samples = self.config.DataParameters.Sampling.num_samples
        self.sampling_info = (
            f"{self.num_samples}_samples" if self.num_samples else "full_data"
        )

        # Gene preprocessing settings
        gene_filtering = self.config.GenePreprocessing.GeneFiltering
        gene_balancing = self.config.GenePreprocessing.GeneBalancing
        config_flags = []

        # Check for hvg or batch_genes options, which are fully separate
        if gene_filtering.highly_variable_genes:
            self.config_subfolder = "hvg"
        elif gene_balancing.balance_genes:
            self.config_subfolder = "balanced_autosomal"
        elif gene_balancing.balance_lnc_genes:
            self.config_subfolder = "balanced_non_lnc"
        elif gene_filtering.select_batch_genes:
            self.config_subfolder = "batch_genes"
        else:
            # Check for gene options
            if gene_filtering.only_keep_lnc_genes:
                config_flags.append("only_lnc")
            if gene_filtering.remove_lnc_genes:
                config_flags.append("no_lnc")
            if gene_filtering.remove_autosomal_genes:
                config_flags.append("no_autosomal")
            if gene_filtering.remove_sex_genes:
                config_flags.append("no_sex")
            if config_flags:
                # Gene options are active and connected
                self.config_subfolder = "_".join(config_flags)
            else:
                # Default to full_data if no options are active
                self.config_subfolder = "full_data"


        # Cell type subfolder
        self.cell_type_folder_name = (
            "all_cells" if self.cell_type == "all" else f"{self.cell_type}"
        )

        # Sex type subfolder with TrainTestSplit handling
        train_test_split = self.config.DataParameters.TrainTestSplit
        self.train_test_split_method = train_test_split.method.lower()

        if self.train_test_split_method == "sex":
            train_sex = train_test_split.train.sex.lower()
            test_sex = train_test_split.test.sex.lower()
            self.subfolder_sex_name = f"train_{train_sex}_test_{test_sex}"
        else:
            self.subfolder_sex_name = (
                "all_sexes" if self.sex_type == "all" else self.sex_type
            )

        # Tissue type subfolder with TrainTestSplit handling
        if self.train_test_split_method == "tissue":
            train_tissue = train_test_split.train.tissue.lower()
            test_tissue = train_test_split.test.tissue.lower()
            self.tissue = f"train_{train_tissue}_test_{test_tissue}"
        else:
            self.tissue = "all_tissues" if self.tissue == "all" else self.tissue

    def construct_model_directory(self):
        """
        Constructs the directory path for the model based on the configuration.

        Returns:
        - str: The path to the model directory.
        """
        # Get the code directory
        code_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the model directory path
        model_dir = os.path.join(
            code_dir,
            "..",
            "Models",
            self.correction_dir,
            self.tissue,
            self.model_type,
            self.encoding_variable,
            self.config_subfolder,
            self.cell_type_folder_name,
            self.subfolder_sex_name,
        )
        return model_dir

    def get_visualization_directory(self, subfolder=None):
        """
        Constructs the output directory path based on the configuration settings.

        Parameters:

        - subfolder (str): Additional subfolder within the analysis directory (e.g., 'SHAP').

        Returns:
        - output_dir (str): The constructed directory path.
        """
        # Get the code directory
        code_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the output directory path
        output_dir = os.path.join(
            code_dir,
            "..",
            "Analysis",
            self.correction_dir,
            self.tissue,
            self.model_type,
            self.encoding_variable,
            self.config_subfolder,
            self.cell_type_folder_name,
            self.subfolder_sex_name,
            "Results",
        )

        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def get_processed_data_dir(self):
        """
        Constructs the directory path for saving or loading preprocessed data based on the configuration.

        Returns:
        - str: The path to the processed data directory.
        """
        # Get the code directory
        code_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the full path
        processed_data_dir = os.path.join(
            code_dir,
            "..",
            "Data",
            self.correction_dir,
            self.tissue,
            self.model_type,
            self.encoding_variable,
            self.config_subfolder,
            self.cell_type_folder_name,
            self.subfolder_sex_name,
        )

        return processed_data_dir
