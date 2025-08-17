"""
Path management utilities for organizing model and data directories.

This module provides comprehensive path management for the TimeFlies project,
implementing the clean 2-level naming convention:
- Level 1: Experiment Type (tissue_model_encoding)  
- Level 2: Configuration Details (method_cells_sexes)

Naming Rules:
- Between main parts: _ (underscores)
- Within compound terms: - (hyphens)

Example: data/processed/uncorrected/head_cnn_age/all-genes_all-cells_all-sexes/
"""

import os
from typing import Any, Optional, Dict
from pathlib import Path


class PathManager:
    """
    A utility class to construct model and data directory paths based on configuration.

    This class centralizes directory path construction logic using the new clean
    2-level naming convention. It automatically generates organized paths for:
    - Processed data storage
    - Model outputs  
    - Analysis results
    - Raw data access

    Attributes:
        config: Configuration object containing experiment parameters
        correction_dir: "batch_corrected" or "uncorrected" 
        experiment_name: Full experiment path (level1/level2)
        base_experiment: Level 1 name (tissue_model_encoding)
        config_details: Level 2 name (method_cells_sexes)
    """

    def __init__(self, config: Any):
        """
        Initialize PathManager with experiment configuration.

        Args:
            config: Configuration object with experiment parameters including:
                - tissue type, model type, encoding variable
                - batch correction settings
                - gene filtering options
                - cell and sex type filters
                - train-test split configuration

        Example:
            config = load_config("configs/head_cnn_age.yaml")
            path_manager = PathManager(config)
            model_dir = path_manager.construct_model_directory()
        """
        self.config = config

        # Extract core parameters with proper error handling
        try:
            # Batch correction setting
            self.batch_correction_enabled = getattr(
                config.data.batch_correction, 'enabled', False
            )
            self.correction_dir = (
                "batch_corrected" if self.batch_correction_enabled else "uncorrected"
            )

            # Core experiment parameters
            self.tissue = getattr(config.data, 'tissue', 'head').lower()
            self.model_type = getattr(config.data, 'model_type', 'CNN').lower()
            self.encoding_variable = getattr(config.data, 'encoding_variable', 'age').lower()
            self.cell_type = getattr(config.data, 'cell_type', 'all').lower()
            self.sex_type = getattr(config.data, 'sex_type', 'all').lower()

            # Generate experiment naming components
            self.experiment_name = self._generate_experiment_name()
            
        except AttributeError as e:
            raise ValueError(f"Invalid configuration structure: {e}")
            
    def _get_project_root(self) -> Path:
        """
        Find project root directory by looking for key directories.
        
        Returns:
            Path: Project root directory
            
        Raises:
            FileNotFoundError: If project root cannot be located
        """
        current_dir = Path(__file__).parent.absolute()
        
        # Search up to 5 levels for project root indicators
        for _ in range(5):
            # Look for key directories that indicate project root
            if any((current_dir / indicator).exists() for indicator in 
                   ["data", "outputs", "src", "configs"]):
                return current_dir
            current_dir = current_dir.parent
            
        raise FileNotFoundError(
            "Could not locate TimeFlies project root. "
            "Make sure you're running from within the project directory."
        )

    def _generate_experiment_name(self) -> str:
        """
        Generate clean experiment name using 2-level naming convention.
        
        Returns:
            str: Full experiment path (level1/level2)
            
        Example:
            head_cnn_age/all-genes_all-cells_all-sexes
        """
        # Level 1: Basic experiment (tissue_model_encoding)
        self.base_experiment = "_".join([
            self.tissue,
            self.model_type, 
            self.encoding_variable
        ])
        
        # Level 2: Configuration details (method_cells_sexes)
        config_parts = [
            self._get_gene_method()
        ]
        
        # Add cell type with proper formatting
        if self.cell_type != "all":
            # Convert spaces to hyphens within cell type names
            cell_type_clean = self.cell_type.replace(" ", "-")
            config_parts.append(cell_type_clean)
        else:
            config_parts.append("all-cells")
            
        # Add sex type 
        if self.sex_type != "all":
            config_parts.append(self.sex_type)
        else:
            config_parts.append("all-sexes")
        
        # Join configuration parts with underscores
        self.config_details = "_".join(config_parts)
        
        # Handle train-test split naming for special cases
        try:
            split_method = getattr(self.config.data.train_test_split, 'method', 'random').lower()
            
            if split_method == "sex":
                train_sex = getattr(self.config.data.train_test_split.train, 'sex', 'male').lower()
                test_sex = getattr(self.config.data.train_test_split.test, 'sex', 'female').lower()
                self.config_details = self.config_details.replace("all-sexes", f"train-{train_sex}_test-{test_sex}")
            elif split_method == "tissue":
                train_tissue = getattr(self.config.data.train_test_split.train, 'tissue', 'head').lower()
                test_tissue = getattr(self.config.data.train_test_split.test, 'tissue', 'body').lower()
                self.base_experiment = self.base_experiment.replace(self.tissue, f"train-{train_tissue}_test-{test_tissue}")
        except AttributeError:
            # Use default naming if train_test_split config is not available
            pass
        
        # Return combined path
        return f"{self.base_experiment}/{self.config_details}"
    
    def _get_gene_method(self) -> str:
        """Get gene preprocessing method name."""
        try:
            gene_filtering = self.config.gene_preprocessing.gene_filtering
            gene_balancing = self.config.gene_preprocessing.gene_balancing
            
            # Check for specific methods
            if getattr(gene_filtering, 'highly_variable_genes', False):
                return "hvg"
            elif getattr(gene_balancing, 'balance_genes', False):
                return "balanced"
            elif getattr(gene_balancing, 'balance_lnc_genes', False):
                return "balanced-lnc"
            elif getattr(gene_filtering, 'select_batch_genes', False):
                return "batch-genes"
            elif getattr(gene_filtering, 'only_keep_lnc_genes', False):
                return "only-lnc"
            elif getattr(gene_filtering, 'remove_lnc_genes', False):
                return "no-lnc"
            elif getattr(gene_filtering, 'remove_autosomal_genes', False):
                return "no-autosomal"
            elif getattr(gene_filtering, 'remove_sex_genes', False):
                return "no-sex"
            else:
                return "all-genes"
        except AttributeError:
            # Fallback if config structure is different
            return "all-genes"

    def construct_model_directory(self) -> str:
        """
        Constructs the directory path for the model based on the configuration.
        
        Returns:
            str: The path to the model directory
            
        Example:
            outputs/models/batch_corrected/head_cnn_age/all-genes_all-cells_all-sexes/
        """
        project_root = self._get_project_root()
        
        # Use clean 2-level structure: outputs/models/correction_dir/level1/level2/
        model_dir = project_root / "outputs" / "models" / self.correction_dir / self.base_experiment / self.config_details
        
        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        return str(model_dir)

    def get_visualization_directory(self, subfolder: Optional[str] = None) -> str:
        """
        Constructs the output directory path for analysis results and visualizations.
        
        Args:
            subfolder: Additional subfolder within the results directory (e.g., 'plots')
            
        Returns:
            str: The constructed directory path
            
        Example:
            outputs/results/batch_corrected/head_cnn_age/all-genes_all-cells_all-sexes/plots/
        """
        project_root = self._get_project_root()
        
        # Use clean 2-level structure: outputs/results/correction_dir/level1/level2/
        output_dir = project_root / "outputs" / "results" / self.correction_dir / self.base_experiment / self.config_details
        
        if subfolder:
            output_dir = output_dir / subfolder
            
        # Create the directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir)

    def get_processed_data_dir(self) -> str:
        """
        Constructs the directory path for saving or loading preprocessed data.
        
        Returns:
            str: The path to the processed data directory
            
        Example:
            data/processed/batch_corrected/head_cnn_age/all-genes_all-cells_all-sexes/
        """
        project_root = self._get_project_root()
        
        # Use clean 2-level structure: data/processed/correction_dir/level1/level2/
        processed_data_dir = project_root / "data" / "processed" / self.correction_dir / self.base_experiment / self.config_details
        
        # Create directory if it doesn't exist
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        return str(processed_data_dir)
    
    def get_raw_data_dir(self, tissue_override: Optional[str] = None) -> str:
        """
        Constructs the directory path for raw h5ad data files.
        
        Args:
            tissue_override: Override the tissue from config (e.g., for cross-tissue analysis)
            
        Returns:
            str: The path to the raw data directory
            
        Example:
            data/raw/h5ad/head/uncorrected/
        """
        project_root = self._get_project_root()
        tissue = tissue_override or self.tissue
        
        # Raw data structure: data/raw/h5ad/tissue/correction_status/
        raw_data_dir = project_root / "data" / "raw" / "h5ad" / tissue / self.correction_dir
        
        return str(raw_data_dir)
    
    def get_log_directory(self) -> str:
        """
        Constructs the directory path for log files.
        
        Returns:
            str: The path to the logs directory
        """
        project_root = self._get_project_root()
        log_dir = project_root / "outputs" / "logs"
        
        # Create directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return str(log_dir)