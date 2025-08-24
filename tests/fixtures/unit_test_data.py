"""Test data creation utilities for unit tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import numpy as np
import pandas as pd


def create_sample_anndata(
    n_cells: int = 100, n_genes: int = 50, add_age: bool = True, add_batch: bool = False
) -> ad.AnnData:
    """Create a sample AnnData object for testing.

    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        add_age: Whether to add age metadata
        add_batch: Whether to add batch metadata

    Returns:
        AnnData object with synthetic data
    """
    # Create synthetic expression data
    np.random.seed(42)  # For reproducible tests
    X = np.random.negative_binomial(10, 0.3, size=(n_cells, n_genes)).astype(float)

    # Create cell metadata
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])

    if add_age:
        # Create age groups (young, middle, old)
        ages = np.random.choice([1, 10, 20], n_cells)
        obs["age"] = ages
        obs["age_group"] = pd.cut(
            ages, bins=[0, 5, 15, 25], labels=["young", "middle", "old"]
        )

        # Add sex and tissue columns for TimeFlies compatibility
        obs["sex"] = np.random.choice(["male", "female"], n_cells)
        obs["tissue"] = np.random.choice(["head", "body"], n_cells)

    if add_batch:
        # Create batch effects
        obs["batch"] = np.random.choice(["batch1", "batch2", "batch3"], n_cells)

    # Create gene metadata
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    var["gene_type"] = np.random.choice(
        ["protein_coding", "lncRNA", "pseudogene"], n_genes
    )
    var["highly_variable"] = np.random.choice([True, False], n_genes, p=[0.2, 0.8])

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)

    return adata


def create_test_project_structure(base_path: Path) -> dict[str, Path]:
    """Create a test project directory structure.

    Args:
        base_path: Base directory path

    Returns:
        Dictionary mapping structure names to paths
    """
    structure = {
        "configs": base_path / "configs",
        "data": base_path / "data",
        "outputs": base_path / "outputs",
        "data_fruitfly_aging": base_path / "data" / "fruitfly_aging" / "head",
        "data_fruitfly_alzheimers": base_path / "data" / "fruitfly_alzheimers" / "head",
        "outputs_fruitfly_aging": base_path / "outputs" / "fruitfly_aging" / "head",
        "outputs_fruitfly_alzheimers": base_path
        / "outputs"
        / "fruitfly_alzheimers"
        / "head",
    }

    # Create all directories
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)

    return structure


def create_minimal_config() -> dict[str, Any]:
    """Create a minimal configuration for testing.

    Returns:
        Dictionary with minimal config settings
    """
    return {
        "project": {"name": "test_project", "tissue": "head", "target": "age"},
        "data": {
            "min_cells": 10,
            "min_genes": 10,
            "max_genes": 5000,
            "mt_cutoff": 20.0,
        },
        "model": {
            "type": "CNN",
            "epochs": 2,  # Small for testing
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "training": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
        "output": {"save_models": True, "save_plots": True},
    }


class TestDataManager:
    """Manager for test data lifecycle."""

    def __init__(self):
        self.temp_dirs = []
        self.test_files = []

    def create_temp_dir(self, prefix: str = "timeflies_test_") -> Path:
        """Create a temporary directory for testing.

        Args:
            prefix: Prefix for temp directory name

        Returns:
            Path to temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_test_h5ad(
        self,
        path: Path,
        n_cells: int = 100,
        n_genes: int = 50,
        add_age: bool = True,
        add_batch: bool = False,
    ) -> Path:
        """Create a test H5AD file.

        Args:
            path: Path where to save the file
            n_cells: Number of cells
            n_genes: Number of genes
            add_age: Whether to add age metadata
            add_batch: Whether to add batch metadata

        Returns:
            Path to created file
        """
        adata = create_sample_anndata(n_cells, n_genes, add_age, add_batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
        self.test_files.append(path)
        return path

    def cleanup(self):
        """Clean up all temporary files and directories."""
        # Remove test files
        for file_path in self.test_files:
            if file_path.exists():
                file_path.unlink()

        # Remove temp directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        # Clear lists
        self.temp_dirs.clear()
        self.test_files.clear()
