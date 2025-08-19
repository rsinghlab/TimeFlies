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
    Create test data by sampling from real TimeFlies data.
    
    Args:
        n_samples: Number of cells to sample
        n_genes: Number of genes to sample
        seed: Random seed for reproducibility
    """
    if not DEPS_AVAILABLE:
        print("Error: Required dependencies (numpy, pandas, scanpy, h5py) not available")
        print("Creating mock test data instead...")
        return create_mock_test_data(n_samples, n_genes, seed)
    
    np.random.seed(seed)
    
    # Path to real data
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "h5ad" / "head"
    
    # Load a small portion of real data
    print(f"Loading real data from {data_dir}")
    
    # Load uncorrected data
    train_path = data_dir / "fly_train.h5ad"
    if not train_path.exists():
        print(f"Error: {train_path} not found")
        print("Creating mock test data instead...")
        return create_mock_test_data(n_samples, n_genes, seed)
        
    print(f"Loading {train_path}")
    adata = sc.read_h5ad(train_path)
    
    print(f"Original data shape: {adata.shape}")
    
    # Sample cells and genes
    cell_indices = np.random.choice(adata.n_obs, min(n_samples, adata.n_obs), replace=False)
    gene_indices = np.random.choice(adata.n_vars, min(n_genes, adata.n_vars), replace=False)
    
    # Create subset
    adata_subset = adata[cell_indices, gene_indices].copy()
    
    print(f"Sampled data shape: {adata_subset.shape}")
    
    # Ensure we have all required metadata
    required_obs_cols = ['age', 'sex', 'tissue', 'dataset', 'afca_annotation_broad']
    for col in required_obs_cols:
        if col not in adata_subset.obs.columns:
            print(f"Warning: Missing column {col}, adding mock data")
            if col == 'age':
                adata_subset.obs[col] = np.random.choice([5, 30, 50, 70], adata_subset.n_obs)
            elif col == 'sex':
                adata_subset.obs[col] = np.random.choice(['male', 'female'], adata_subset.n_obs)
            elif col == 'tissue':
                adata_subset.obs[col] = 'head'
            elif col == 'dataset':
                adata_subset.obs[col] = 'test_dataset'
            elif col == 'afca_annotation_broad':
                adata_subset.obs[col] = np.random.choice(
                    ['CNS neuron', 'muscle cell', 'epithelial cell'], 
                    adata_subset.n_obs
                )
    
    # Save test data
    output_dir = Path(__file__).parent
    output_path = output_dir / "test_data.h5ad"
    
    print(f"Saving test data to {output_path}")
    adata_subset.write_h5ad(output_path)
    
    # Also create a batch-corrected version (just copy with a layer added)
    adata_corrected = adata_subset.copy()
    # Simulate scVI normalization (just log-transform for testing)
    adata_corrected.layers['scvi_normalized'] = np.log1p(adata_corrected.X)
    
    output_path_corrected = output_dir / "test_data_corrected.h5ad"
    print(f"Saving corrected test data to {output_path_corrected}")
    adata_corrected.write_h5ad(output_path_corrected)
    
    # Create summary statistics
    stats = {
        'n_obs': adata_subset.n_obs,
        'n_vars': adata_subset.n_vars,
        'age_distribution': adata_subset.obs['age'].value_counts().to_dict(),
        'sex_distribution': adata_subset.obs['sex'].value_counts().to_dict(),
        'cell_types': adata_subset.obs['afca_annotation_broad'].value_counts().to_dict() if 'afca_annotation_broad' in adata_subset.obs else {},
        'data_range': {
            'min': float(np.min(adata_subset.X)),
            'max': float(np.max(adata_subset.X)),
            'mean': float(np.mean(adata_subset.X)),
            'sparsity': float(np.mean(adata_subset.X == 0))
        }
    }
    
    # Save stats
    stats_path = output_dir / "test_data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Saved statistics to {stats_path}")
    print("\nTest data statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    return adata_subset, adata_corrected


def create_mock_test_data(n_samples=100, n_genes=500, seed=42):
    """
    Create mock test data when real data or dependencies aren't available.
    """
    print(f"Creating mock test data: {n_samples} samples, {n_genes} genes")
    
    # Create mock statistics to save
    stats = {
        'n_obs': n_samples,
        'n_vars': n_genes,
        'age_distribution': {'5': 25, '30': 35, '50': 25, '70': 15},
        'sex_distribution': {'male': 50, 'female': 50},
        'cell_types': {'CNS neuron': 40, 'muscle cell': 30, 'epithelial cell': 30},
        'data_range': {
            'min': 0.0,
            'max': 100.0,
            'mean': 5.2,
            'sparsity': 0.85
        },
        'mock_data': True
    }
    
    # Save stats
    output_dir = Path(__file__).parent
    stats_path = output_dir / "test_data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Saved mock statistics to {stats_path}")
    print("\nMock test data statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Create placeholder files to indicate mock data was created
    mock_files = [
        output_dir / "test_data.h5ad.mock",
        output_dir / "test_data_corrected.h5ad.mock"
    ]
    
    for mock_file in mock_files:
        with open(mock_file, 'w') as f:
            f.write(f"Mock file placeholder - real data requires: numpy, pandas, scanpy, h5py\n")
            f.write(f"Created: {n_samples} samples, {n_genes} genes, seed: {seed}\n")
        print(f"Created mock placeholder: {mock_file}")
    
    return None

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