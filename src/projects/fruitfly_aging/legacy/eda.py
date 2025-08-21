# eda_handler.py

import pandas as pd
import numpy as np
from scipy import stats
import scipy
from visuals import VisualizationTools


class EDAHandler:
    """
    A class to handle Exploratory Data Analysis (EDA) on multiple datasets.

    This class takes in multiple datasets and performs EDA on them, leveraging
    configuration settings to customize the analysis.
    """

    def __init__(
        self,
        config,
        path_manager,
        adata,
        adata_eval,
        adata_original,
        adata_corrected,
        adata_eval_corrected,
    ):
        """
        Initializes the EDAHandler with the given configuration and datasets.

        Parameters:
        - config (ConfigHandler): Configuration settings for the EDA.
        - path_manager (PathManager): Path manager object to handle directory paths.
        - adata (AnnData): Main AnnData object containing the dataset.
        - adata_eval (AnnData): AnnData object for evaluation data.
        - adata_original (AnnData): The original unfiltered AnnData object.
        - adata_corrected (AnnData): The batch-corrected AnnData object.
        - adata_eval_corrected (AnnData): The batch-corrected evaluation AnnData object.
        """
        self.config = config
        self.path_manager = path_manager
        self.adata = adata
        self.adata_eval = adata_eval
        self.adata_original = adata_original
        self.adata_corrected = adata_corrected
        self.adata_eval_corrected = adata_eval_corrected

        # Initialize VisualizationTools with config and path_manager
        self.visual_tools = VisualizationTools(
            config=self.config, path_manager=self.path_manager
        )

    def _print_general_info(self, adata):
        """
        Print general information about the dataset including its shape,
        variable names, and columns in 'obs'.

        Args:
            adata (AnnData): The input dataset.
        """
        print(f"\nData Shape: {adata.shape}")
        print(f"\nVariable names: {adata.var_names}")
        print(f"\nColumns in 'obs': {adata.obs.columns}")

    def _display_class_info(self, adata, encoding_column, folder_name, dataset_name):
        """
        Display information about the unique classes and class counts in the dataset.

        Args:
            adata (AnnData): The input dataset.
            encoding_column (str): Column used for encoding class labels.
            folder_name (str): Name of the folder where visualizations will be saved.
            dataset_name (str): Name of the dataset for labeling outputs.
        """
        unique_classes = np.unique(adata.obs[encoding_column])
        class_mapping = {
            class_name: f"Class {idx}" for idx, class_name in enumerate(unique_classes)
        }

        print("\nUnique Classes:")
        for class_name in unique_classes:
            print(f"{class_mapping[class_name]}: {class_name}")

        num_classes = len(unique_classes)
        print(f"\nNumber of unique classes: {num_classes}")

        # Class distribution
        class_counts = adata.obs[encoding_column].value_counts().sort_values()
        print(f"\nClass Counts:\n{class_counts}")

        # Plot class distribution
        self.visual_tools.plot_class_distribution(
            class_counts=class_counts,
            file_name=f"{dataset_name}_class_distribution.png",
            dataset=folder_name,
            subfolder_name="EDA",
        )

    def _analyze_gene_statistics(self, adata, folder_name, dataset_name):
        """
        Perform statistical analysis and outlier detection for gene expression data.

        Args:
            adata (AnnData): The input dataset.
            folder_name (str): Name of the folder where visualizations will be saved.
            dataset_name (str): Name of the dataset for labeling outputs.
        """
        # Statistical summary of the first 10 genes
        data_arr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        stats_df = self._generate_stat_summary(data_arr, adata.var_names)
        print(f"\nStatistical Summary for the first 10 genes:\n{stats_df}")

        # Outlier detection for the first 10 genes
        num_outliers = self._detect_outliers(data_arr, threshold=3)
        print(f"\nNumber of outliers in the gene expression data: {num_outliers}")

        # Visualize gene expression overview
        gene_expression_matrix = data_arr[:5, :5]
        gene_expression_df = pd.DataFrame(
            data=gene_expression_matrix,
            index=adata.obs.index[:5],
            columns=adata.var.index[:5],
        )
        self.visual_tools.create_styled_dataframe(
            df=gene_expression_df,
            subfolder_name="EDA",
            dataset=folder_name,
            file_name=f"{dataset_name}_sampled_gene_expression_overview.png",
        )

        self.visual_tools.create_styled_dataframe(
            df=stats_df,
            subfolder_name="EDA",
            dataset=folder_name,
            file_name=f"{dataset_name}_gene_expression_statistics.png",
        )

    def _handle_missing_values(self, adata):
        """
        Calculate and return the number of missing values in the dataset.

        Args:
            adata (AnnData): Input AnnData object.

        Returns:
            int: Number of missing values in the dataset.
        """
        if scipy.sparse.issparse(adata.X):
            data_arr = adata.X.toarray()
        else:
            data_arr = adata.X
        return np.isnan(data_arr).sum()

    def _check_for_duplicates(self, df):
        """
        Checks for duplicate rows in the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            num_duplicates (int): Number of duplicate rows.
            duplicate_rows (pd.DataFrame): DataFrame of duplicate rows.
        """
        duplicate_rows = df.duplicated()
        num_duplicates = duplicate_rows.sum()
        return num_duplicates, df[duplicate_rows]

    def _detect_outliers(self, data, threshold=3):
        """
        Detects outliers in the given data based on the z-score.

        Args:
            data (np.array): Input data.
            threshold (float): The z-score threshold to define outliers.

        Returns:
            int: Number of outliers detected.
        """
        z_scores = np.abs(stats.zscore(data[:, :10], axis=0))
        return np.where(z_scores > threshold)[0].shape[0]

    def _generate_stat_summary(self, data, var_names, num_genes=10):
        """
        Generates a statistical summary for the first few genes.

        Args:
            data (np.array): Gene expression data.
            var_names (pd.Index): Variable names (gene names) corresponding to the data columns.
            num_genes (int): The number of genes to include in the summary (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing the statistical summary for the specified number of genes.
        """
        # Generate a statistical summary for the first `num_genes` genes
        stats_df = (
            pd.DataFrame(data[:, :num_genes], columns=var_names[:num_genes])
            .describe()
            .transpose()
        )
        return stats_df

    def eda(self, adata, encoding_column, folder_name, dataset_name):
        """
        Perform exploratory data analysis on the dataset.

        Args:
            adata (AnnData): The input data for analysis.
            encoding_column (str): Column used for encoding class labels.
            folder_name (str): Name of the folder where visualizations will be saved.
            dataset_name (str): Name of the dataset for labeling outputs.

        Returns:
            None
        """
        # Display the first 5 rows of the 'obs' dataframe
        print("\nFirst 5 rows of 'obs' dataframe:")
        print(adata.obs.head(5))

        # Visualize the 'obs' dataframe
        self.visual_tools.create_styled_dataframe(
            df=adata.obs.head(5),
            subfolder_name="EDA",
            dataset=folder_name,
            file_name=f"{dataset_name}_observation_df_overview.png",
        )

        # Print and visualize basic dataset information
        self._print_general_info(adata)

        # Visualize sparse vs dense matrices
        self.visual_tools.visualize_sparse_vs_dense(
            adata=adata,
            subset_size=50,
            head_size=5,
            file_name=f"{dataset_name}_sparse_vs_dense.png",
            dataset=folder_name,
            subfolder_name="EDA",
        )

        # Handle duplicate rows using a utility function
        adata.obs = adata.obs.reset_index(drop=False)
        num_duplicates, duplicate_rows = self._check_for_duplicates(adata.obs)
        print(f"\nNumber of duplicate rows in 'obs': {num_duplicates}")
        if num_duplicates > 0:
            print("Duplicate rows in 'obs':")
            print(duplicate_rows)
        adata.obs.set_index("index", inplace=True)

        # Handle missing values using a utility function
        missing_values = self._handle_missing_values(adata)
        print(f"\nMissing Values: {missing_values}")

        # Display unique classes and class counts
        self._display_class_info(
            adata=adata,
            encoding_column=encoding_column,
            folder_name=folder_name,
            dataset_name=dataset_name,
        )

        # Perform statistical analysis and outlier detection
        self._analyze_gene_statistics(
            adata=adata, folder_name=folder_name, dataset_name=dataset_name
        )

    def run_eda(self):
        """
        Runs EDA based on the configuration settings, deciding whether to
        analyze corrected or uncorrected datasets, depending on the batch correction.

        Returns:
            None
        """
        if self.config.DataParameters.BatchCorrection.enabled:
            # Run EDA on batch-corrected datasets
            self.eda(
                adata=self.adata_corrected,
                encoding_column=self.config.DataParameters.GeneralSettings.encoding_variable,
                folder_name="Batch Training Data",
                dataset_name="batch_train",
            )
            self.eda(
                adata=self.adata_eval_corrected,
                encoding_column=self.config.DataParameters.GeneralSettings.encoding_variable,
                folder_name="Batch Evaluation Data",
                dataset_name="batch_evaluation",
            )
        else:
            # Run EDA on uncorrected datasets
            self.eda(
                adata=self.adata,
                encoding_column=self.config.DataParameters.GeneralSettings.encoding_variable,
                folder_name="Training Data",
                dataset_name="train",
            )
            self.eda(
                adata=self.adata_eval,
                encoding_column=self.config.DataParameters.GeneralSettings.encoding_variable,
                folder_name="Evaluation Data",
                dataset_name="evaluation",
            )
            self.eda(
                adata=self.adata_original,
                encoding_column=self.config.DataParameters.GeneralSettings.encoding_variable,
                folder_name="Original Data",
                dataset_name="original",
            )
