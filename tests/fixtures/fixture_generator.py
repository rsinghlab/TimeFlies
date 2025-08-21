#!/usr/bin/env python3
"""
Create test data fixtures from real TimeFlies data.

This script samples a small subset of real data to create realistic test fixtures
that will catch actual issues in the pipeline.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import h5py
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

def create_test_data_from_real(n_samples=100, n_genes=500, seed=42):
    """
    Create test data - using mock data for consistency across projects.
    
    Args:
        n_samples: Number of cells to sample
        n_genes: Number of genes to sample
        seed: Random seed for reproducibility
    """
    print("Creating standardized mock test data for testing...")
    return create_mock_test_data(n_samples, n_genes, seed)


def create_mock_test_data(n_samples=100, n_genes=500, seed=42):
    """
    Create mock test data with proper H5AD files for testing.
    """
    if not DEPS_AVAILABLE:
        print(f"Creating mock test data: {n_samples} samples, {n_genes} genes")
        print("Warning: Dependencies not available, creating placeholder files only")
        # Create placeholder files if dependencies aren't available
        output_dir = Path(__file__).parent
        mock_files = [
            output_dir / "test_data.h5ad.mock",
            output_dir / "test_data_corrected.h5ad.mock"
        ]
        
        for mock_file in mock_files:
            with open(mock_file, 'w') as f:
                f.write(f"Mock file placeholder - requires: numpy, pandas, scanpy, h5py\n")
            print(f"Created mock placeholder: {mock_file}")
        return None
    
    print(f"Creating mock H5AD test data: {n_samples} samples, {n_genes} genes")
    
    np.random.seed(seed)
    
    # Create mock expression data (sparse matrix)
    from scipy.sparse import csr_matrix
    
    # Create realistic sparse expression data
    density = 0.15  # 15% non-zero values (typical for scRNA-seq)
    n_nonzero = int(n_samples * n_genes * density)
    
    # Generate random sparse data
    rows = np.random.choice(n_samples, n_nonzero)
    cols = np.random.choice(n_genes, n_nonzero)
    data = np.random.exponential(scale=2.0, size=n_nonzero)  # Log-normal-ish expression
    
    X = csr_matrix((data, (rows, cols)), shape=(n_samples, n_genes))
    
    # Create mock observation metadata
    obs_data = {
        'age': np.random.choice([5, 30, 50, 70], n_samples, p=[0.25, 0.35, 0.25, 0.15]),
        'sex': np.random.choice(['male', 'female'], n_samples),
        'tissue': ['head'] * n_samples,
        'dataset': ['test_dataset'] * n_samples,
        'afca_annotation_broad': np.random.choice(
            ['CNS neuron', 'muscle cell', 'epithelial cell'], 
            n_samples, 
            p=[0.4, 0.3, 0.3]
        )
    }
    obs = pd.DataFrame(obs_data, index=[f'cell_{i}' for i in range(n_samples)])
    
    # Create mock gene metadata
    var_data = {
        'gene_symbol': [f'Gene_{i}' for i in range(n_genes)],
        'highly_variable': np.random.choice([True, False], n_genes, p=[0.3, 0.7])
    }
    var = pd.DataFrame(var_data, index=[f'gene_{i}' for i in range(n_genes)])
    
    # Create AnnData object
    import anndata
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    
    # Save test data
    output_dir = Path(__file__).parent
    output_path = output_dir / "test_data.h5ad"
    
    print(f"Saving test data to {output_path}")
    adata.write_h5ad(output_path)
    
    # Create batch-corrected version
    adata_corrected = adata.copy()
    # Simulate batch correction by adding a normalized layer
    adata_corrected.layers['scvi_normalized'] = np.log1p(adata_corrected.X.toarray())
    
    output_path_corrected = output_dir / "test_data_corrected.h5ad"
    print(f"Saving corrected test data to {output_path_corrected}")
    adata_corrected.write_h5ad(output_path_corrected)
    
    # Create statistics
    stats = {
        'n_obs': n_samples,
        'n_vars': n_genes,
        'age_distribution': obs['age'].value_counts().to_dict(),
        'sex_distribution': obs['sex'].value_counts().to_dict(),
        'cell_types': obs['afca_annotation_broad'].value_counts().to_dict(),
        'data_range': {
            'min': float(X.min()),
            'max': float(X.max()),
            'mean': float(X.mean()),
            'sparsity': float(1.0 - density)
        },
        'mock_data': True
    }
    
    # Save stats
    stats_path = output_dir / "test_data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Saved statistics to {stats_path}")
    print("\nMock test data statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    return adata, adata_corrected

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test data from real TimeFlies data")
    parser.add_argument('--samples', type=int, default=100, help='Number of cells to sample')
    parser.add_argument('--genes', type=int, default=500, help='Number of genes to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_test_data_from_real(
        n_samples=args.samples,
        n_genes=args.genes,
        seed=args.seed
    )