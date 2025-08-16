"""Path management utilities for organizing model and data directories."""

import os
from typing import Any, Optional


class PathManager:
    """
    A utility class to construct model and visualization directory paths based on the configuration.

    This class centralizes the directory path construction logic to avoid code duplication
    between different components of the system.
    """

    def __init__(self, config: Any):
        """
        Initializes the PathManager with the given configuration.

        Parameters:
        - config: The configuration object containing all necessary parameters.
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

        # Generate clean experiment name
        self.experiment_name = self._generate_experiment_name()

    def _generate_experiment_name(self) -> str:
        """Generate clean experiment name from configuration."""
        # Level 1: Basic experiment (tissue-model-encoding)
        self.base_experiment = "-".join([
            self.tissue,
            self.model_type.lower(), 
            self.encoding_variable
        ])
        
        # Level 2: Configuration details (gene_method-cells-sex)
        config_parts = [
            self._get_gene_method()
        ]
        
        # Add cell type
        if self.cell_type != "all":
            cell_type_clean = self.cell_type.replace(" ", "_")
            config_parts.append(cell_type_clean)
        else:
            config_parts.append("all_cells")
            
        # Add sex type 
        if self.sex_type != "all":
            config_parts.append(self.sex_type)
        else:
            config_parts.append("all_sexes")
        
        self.config_details = "-".join(config_parts)
        
        # Handle train-test split naming
        train_test_split = self.config.DataParameters.TrainTestSplit
        split_method = train_test_split.method.lower()
        
        if split_method == "sex":
            train_sex = train_test_split.train.sex.lower()
            test_sex = train_test_split.test.sex.lower()
            self.config_details = self.config_details.replace("all_sexes", f"train_{train_sex}-test_{test_sex}")
        elif split_method == "tissue":
            train_tissue = train_test_split.train.tissue.lower()
            test_tissue = train_test_split.test.tissue.lower()
            self.base_experiment = self.base_experiment.replace(self.tissue, f"train_{train_tissue}-test_{test_tissue}")
        
        # Return combined for compatibility, but store both parts
        return f"{self.base_experiment}/{self.config_details}"
    
    def _get_gene_method(self) -> str:
        """Get gene preprocessing method name."""
        gene_filtering = self.config.GenePreprocessing.GeneFiltering
        gene_balancing = self.config.GenePreprocessing.GeneBalancing
        
        # Check for specific methods
        if gene_filtering.highly_variable_genes:
            return "hvg"
        elif gene_balancing.balance_genes:
            return "balanced"
        elif gene_balancing.balance_lnc_genes:
            return "balanced_lnc"
        elif gene_filtering.select_batch_genes:
            return "batch_genes"
        elif gene_filtering.only_keep_lnc_genes:
            return "lnc_only"
        elif gene_filtering.remove_lnc_genes:
            return "no_lnc"
        elif gene_filtering.remove_autosomal_genes:
            return "no_autosomal"
        elif gene_filtering.remove_sex_genes:
            return "no_sex"
        else:
            return "full"

    def construct_model_directory(self) -> str:
        """
        Constructs the directory path for the model based on the configuration.

        Returns:
        - str: The path to the model directory.
        """
        # Find project root (look for outputs directory, create if not exists)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        
        # Navigate up to find project root
        for _ in range(5):  # Max 5 levels up
            if (os.path.exists(os.path.join(project_root, "outputs")) or 
                os.path.exists(os.path.join(project_root, "Models"))):
                break
            project_root = os.path.dirname(project_root)
        
        # Use new clean structure: outputs/models/batch_corrected/head_cnn_age/hvg_all_cells_all_sexes/
        model_dir = os.path.join(
            project_root,
            "outputs",
            "models", 
            self.correction_dir,
            self.base_experiment,
            self.config_details
        )
        return model_dir

    def get_visualization_directory(self, subfolder: Optional[str] = None) -> str:
        """
        Constructs the output directory path based on the configuration settings.

        Parameters:
        - subfolder: Additional subfolder within the results directory (e.g., 'plots').

        Returns:
        - output_dir: The constructed directory path.
        """
        # Find project root 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        
        # Navigate up to find project root
        for _ in range(5):  # Max 5 levels up
            if (os.path.exists(os.path.join(project_root, "outputs")) or
                os.path.exists(os.path.join(project_root, "Analysis"))):
                break
            project_root = os.path.dirname(project_root)

        # Use new clean structure: outputs/results/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/
        output_dir = os.path.join(
            project_root,
            "outputs",
            "results",
            self.correction_dir,
            self.base_experiment,
            self.config_details
        )

        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def get_processed_data_dir(self) -> str:
        """
        Constructs the directory path for saving or loading preprocessed data based on the configuration.

        Returns:
        - str: The path to the processed data directory.
        """
        # Find project root 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        
        # Navigate up to find project root
        for _ in range(5):  # Max 5 levels up
            if (os.path.exists(os.path.join(project_root, "data")) or
                os.path.exists(os.path.join(project_root, "Data"))):
                break
            project_root = os.path.dirname(project_root)

        # Use new clean structure: data/processed/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/
        processed_data_dir = os.path.join(
            project_root,
            "data",
            "processed",
            self.correction_dir,
            self.base_experiment,
            self.config_details
        )

        return processed_data_dir