"""Sample data fixtures for testing."""

import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import os
from pathlib import Path


def create_sample_anndata(n_obs=1000, n_vars=2000, tissue="head"):
    """
    Create sample AnnData object for testing.
    
    Args:
        n_obs: Number of observations (cells)
        n_vars: Number of variables (genes)
        tissue: Tissue type
        
    Returns:
        AnnData object with realistic single-cell structure
    """
    # Create random expression data
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create gene names
    var_names = [f"Gene_{i}" for i in range(n_vars)]
    
    # Create cell metadata
    obs_data = {
        "age": np.random.choice([1, 5, 10, 20], n_obs),
        "sex": np.random.choice(["male", "female"], n_obs),
        "tissue": [tissue] * n_obs,
        "dataset": np.random.choice(["dataset_1", "dataset_2", "dataset_3"], n_obs),
        "afca_annotation_broad": np.random.choice([
            "CNS neuron", "sensory neuron", "epithelial cell", 
            "muscle cell", "glial cell"
        ], n_obs),
        "n_genes": np.random.randint(500, 3000, n_obs),
        "total_counts": np.random.randint(1000, 10000, n_obs)
    }
    
    # Create gene metadata
    var_data = {
        "gene_type": np.random.choice(["protein_coding", "lncRNA", "pseudogene"], n_vars),
        "chromosome": np.random.choice([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"], n_vars),
        "highly_variable": np.random.choice([True, False], n_vars, p=[0.2, 0.8])
    }
    
    # Create AnnData object
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(var_data, index=var_names)
    )
    
    # Add some layers
    adata.layers["counts"] = adata.X.copy()
    adata.layers["logcounts"] = np.log1p(adata.X)
    
    return adata


def create_sample_h5ad_files(output_dir, tissues=["head", "body"]):
    """
    Create sample h5ad files for testing.
    
    Args:
        output_dir: Directory to save files
        tissues: List of tissues to create
        
    Returns:
        Dictionary mapping file paths to created files
    """
    output_dir = Path(output_dir)
    file_paths = {}
    
    for tissue in tissues:
        # Create tissue directory structure
        tissue_dir = output_dir / tissue
        uncorrected_dir = tissue_dir / "uncorrected"
        batch_corrected_dir = tissue_dir / "batch_corrected"
        
        uncorrected_dir.mkdir(parents=True, exist_ok=True)
        batch_corrected_dir.mkdir(parents=True, exist_ok=True)
        
        # Create different sized datasets
        adata_original = create_sample_anndata(n_obs=5000, tissue=tissue)
        adata_train = create_sample_anndata(n_obs=3000, tissue=tissue)
        adata_eval = create_sample_anndata(n_obs=1000, tissue=tissue)
        
        # Save uncorrected files
        original_path = uncorrected_dir / "fly_original.h5ad"
        train_path = uncorrected_dir / "fly_train.h5ad"
        eval_path = uncorrected_dir / "fly_eval.h5ad"
        
        adata_original.write_h5ad(original_path)
        adata_train.write_h5ad(train_path)
        adata_eval.write_h5ad(eval_path)
        
        file_paths[f"{tissue}_original"] = str(original_path)
        file_paths[f"{tissue}_train"] = str(train_path)
        file_paths[f"{tissue}_eval"] = str(eval_path)
        
        # Create batch-corrected versions (just copies for testing)
        batch_original_path = batch_corrected_dir / "fly_original_batch.h5ad"
        batch_train_path = batch_corrected_dir / "fly_train_batch.h5ad"
        batch_eval_path = batch_corrected_dir / "fly_eval_batch.h5ad"
        
        # Add mock scVI latent representation
        adata_original.obsm["X_scVI"] = np.random.normal(0, 1, (adata_original.n_obs, 25))
        adata_train.obsm["X_scVI"] = np.random.normal(0, 1, (adata_train.n_obs, 25))
        adata_eval.obsm["X_scVI"] = np.random.normal(0, 1, (adata_eval.n_obs, 25))
        
        adata_original.write_h5ad(batch_original_path)
        adata_train.write_h5ad(batch_train_path)
        adata_eval.write_h5ad(batch_eval_path)
        
        file_paths[f"{tissue}_batch_original"] = str(batch_original_path)
        file_paths[f"{tissue}_batch_train"] = str(batch_train_path)
        file_paths[f"{tissue}_batch_eval"] = str(batch_eval_path)
    
    return file_paths


def create_sample_gene_lists(output_dir):
    """
    Create sample gene list files.
    
    Args:
        output_dir: Directory to save gene lists
        
    Returns:
        Dictionary of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create autosomal genes list
    autosomal_genes = pd.DataFrame({
        "gene_name": [f"Gene_{i}" for i in range(100, 200)]
    })
    autosomal_path = output_dir / "autosomal.csv"
    autosomal_genes.to_csv(autosomal_path, index=False)
    
    # Create sex genes list
    sex_genes = pd.DataFrame({
        "gene_name": [f"Gene_{i}" for i in range(200, 250)]
    })
    sex_path = output_dir / "sex.csv"
    sex_genes.to_csv(sex_path, index=False)
    
    return {
        "autosomal": str(autosomal_path),
        "sex": str(sex_path)
    }


def create_test_project_structure(base_dir):
    """
    Create complete test project structure with sample data.
    
    Args:
        base_dir: Base directory for test project
        
    Returns:
        Dictionary with all created file paths
    """
    base_dir = Path(base_dir)
    
    # Create directory structure
    directories = [
        "data/raw/h5ad",
        "data/raw/gene_lists", 
        "data/processed",
        "outputs/models",
        "outputs/results",
        "outputs/logs",
        "src/projects",
        "src/shared",
        "configs",
        "tests"
    ]
    
    for dir_path in directories:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create sample data files
    h5ad_files = create_sample_h5ad_files(base_dir / "data" / "raw" / "h5ad")
    gene_files = create_sample_gene_lists(base_dir / "data" / "raw" / "gene_lists")
    
    # Create sample config
    config_data = {
        "general": {"project_name": "test_project", "random_state": 42},
        "data": {
            "tissue": "head",
            "model_type": "CNN",
            "target_variable": "age",
            "cell_type": "all",
            "sex_type": "all",
            "batch_correction": {"enabled": False},
            "train_test_split": {"method": "random", "test_split": 0.2}
        },
        "model": {"training": {"epochs": 5, "batch_size": 32}},
        "gene_preprocessing": {
            "gene_filtering": {"highly_variable_genes": False},
            "gene_balancing": {"balance_genes": False}
        }
    }
    
    import yaml
    config_path = base_dir / "configs" / "test.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return {
        "base_dir": str(base_dir),
        "config": str(config_path),
        **h5ad_files,
        **gene_files
    }


class TestDataManager:
    """Context manager for test data creation and cleanup."""
    
    def __init__(self, create_full_structure=True):
        self.create_full_structure = create_full_structure
        self.temp_dir = None
        self.file_paths = {}
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        
        if self.create_full_structure:
            self.file_paths = create_test_project_structure(self.temp_dir)
        else:
            # Just create minimal structure
            base_dir = Path(self.temp_dir)
            base_dir.mkdir(exist_ok=True)
            self.file_paths = {"base_dir": str(base_dir)}
            
        return self.file_paths
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)


# Convenient functions for pytest fixtures
def create_minimal_config():
    """Create minimal valid configuration for testing."""
    return {
        "general": {"project_name": "test", "random_state": 42},
        "data": {
            "tissue": "head",
            "model_type": "CNN", 
            "target_variable": "age",
            "cell_type": "all",
            "sex_type": "all",
            "batch_correction": {"enabled": False}
        },
        "model": {"training": {"epochs": 1}},
        "gene_preprocessing": {
            "gene_filtering": {"highly_variable_genes": False},
            "gene_balancing": {"balance_genes": False}
        }
    }