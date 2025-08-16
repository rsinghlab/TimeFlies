"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
import scanpy as sc
from anndata import AnnData

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from timeflies.data.loaders import DataLoader
from timeflies.data.preprocessing.gene_filter import GeneFilter
from timeflies.data.preprocessing.data_processor import DataPreprocessor
from timeflies.utils.exceptions import DataError


class TestDataLoader:
    """Test data loading utilities."""
    
    def setUp(self):
        """Set up mock configuration."""
        self.mock_config = MagicMock()
        self.mock_config.DataParameters.GeneralSettings.tissue = "head"
        self.mock_config.FileLocations.training_file = "train.h5ad"
        self.mock_config.FileLocations.evaluation_file = "eval.h5ad"
        self.mock_config.FileLocations.original_file = "original.h5ad"
        self.mock_config.FileLocations.batch_corrected_files.train = "train_batch.h5ad"
        self.mock_config.FileLocations.batch_corrected_files.eval = "eval_batch.h5ad"
    
    @patch('scanpy.read_h5ad')
    def test_load_data(self, mock_read_h5ad):
        """Test basic data loading."""
        self.setUp()
        
        # Mock AnnData objects
        mock_adata = MagicMock(spec=AnnData)
        mock_read_h5ad.return_value = mock_adata
        
        loader = DataLoader(self.mock_config)
        adata, adata_eval, adata_original = loader.load_data()
        
        assert adata == mock_adata
        assert adata_eval == mock_adata
        assert adata_original == mock_adata
        assert mock_read_h5ad.call_count == 3
    
    @patch('scanpy.read_h5ad')
    def test_load_corrected_data(self, mock_read_h5ad):
        """Test batch-corrected data loading."""
        self.setUp()
        
        mock_adata = MagicMock(spec=AnnData)
        mock_read_h5ad.return_value = mock_adata
        
        loader = DataLoader(self.mock_config)
        adata_corrected, adata_eval_corrected = loader.load_corrected_data()
        
        assert adata_corrected == mock_adata
        assert adata_eval_corrected == mock_adata
        assert mock_read_h5ad.call_count == 2
    
    @patch('pandas.read_csv')
    def test_load_gene_lists(self, mock_read_csv):
        """Test gene list loading."""
        self.setUp()
        
        # Mock CSV data
        mock_df = pd.DataFrame({'0': ['gene1', 'gene2', 'gene3']})
        mock_read_csv.return_value = mock_df
        
        loader = DataLoader(self.mock_config)
        autosomal_genes, sex_genes = loader.load_gene_lists()
        
        assert autosomal_genes == ['gene1', 'gene2', 'gene3']
        assert sex_genes == ['gene1', 'gene2', 'gene3']
        assert mock_read_csv.call_count == 2


class TestGeneFilter:
    """Test gene filtering utilities."""
    
    def setUp(self):
        """Set up test data and configuration."""
        # Create mock AnnData objects
        n_obs, n_vars = 100, 1000
        X = np.random.randn(n_obs, n_vars)
        
        # Create gene names with some lncRNA genes
        var_names = [f"gene_{i}" for i in range(900)]
        var_names.extend([f"lnc_gene_{i}" for i in range(100)])
        
        obs_data = pd.DataFrame({
            'sex': np.random.choice(['male', 'female'], n_obs),
            'tissue': np.random.choice(['head', 'body'], n_obs),
            'age': np.random.choice(['young', 'old'], n_obs)
        })
        
        var_data = pd.DataFrame(index=var_names)
        
        self.adata = AnnData(X=X, obs=obs_data, var=var_data)
        self.adata_eval = self.adata.copy()
        self.adata_original = self.adata.copy()
        
        # Mock gene lists
        self.autosomal_genes = [f"gene_{i}" for i in range(500)]
        self.sex_genes = [f"gene_{i}" for i in range(500, 600)]
        
        # Mock configuration
        self.mock_config = MagicMock()
        self.mock_config.DataSplit.random_state = 42
        self.mock_config.GenePreprocessing.GeneFiltering.remove_unaccounted_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.remove_sex_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.remove_autosomal_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.only_keep_lnc_genes = False
        self.mock_config.GenePreprocessing.GeneFiltering.remove_lnc_genes = False
        self.mock_config.GenePreprocessing.GeneBalancing.balance_genes = False
        self.mock_config.GenePreprocessing.GeneBalancing.balance_lnc_genes = False
    
    def test_gene_filter_initialization(self):
        """Test GeneFilter initialization."""
        self.setUp()
        
        gene_filter = GeneFilter(
            self.mock_config,
            self.adata,
            self.adata_eval,
            self.adata_original,
            self.autosomal_genes,
            self.sex_genes
        )
        
        assert gene_filter.config == self.mock_config
        assert gene_filter.adata is self.adata
        assert gene_filter.autosomal_genes == self.autosomal_genes
        assert gene_filter.sex_genes == self.sex_genes
    
    def test_balance_genes(self):
        """Test gene balancing functionality."""
        self.setUp()
        
        gene_filter = GeneFilter(
            self.mock_config,
            self.adata,
            self.adata_eval,
            self.adata_original,
            self.autosomal_genes,
            self.sex_genes
        )
        
        # Test balancing with more genes in first list
        gene_list_1 = [f"gene_{i}" for i in range(100)]
        gene_list_2 = [f"gene_{i}" for i in range(50)]
        
        balanced = gene_filter.balance_genes(gene_list_1, gene_list_2)
        
        assert len(balanced) == len(gene_list_2)
        assert all(gene in gene_list_1 for gene in balanced)
    
    def test_balance_genes_insufficient(self):
        """Test gene balancing with insufficient genes."""
        self.setUp()
        
        gene_filter = GeneFilter(
            self.mock_config,
            self.adata,
            self.adata_eval,
            self.adata_original,
            self.autosomal_genes,
            self.sex_genes
        )
        
        # Test with insufficient genes
        gene_list_1 = [f"gene_{i}" for i in range(10)]
        gene_list_2 = [f"gene_{i}" for i in range(50)]
        
        with pytest.raises(ValueError):
            gene_filter.balance_genes(gene_list_1, gene_list_2)
    
    def test_create_and_apply_mask_lnc_only(self):
        """Test filtering to keep only lncRNA genes."""
        self.setUp()
        self.mock_config.GenePreprocessing.GeneFiltering.only_keep_lnc_genes = True
        
        gene_filter = GeneFilter(
            self.mock_config,
            self.adata,
            self.adata_eval,
            self.adata_original,
            self.autosomal_genes,
            self.sex_genes
        )
        
        filtered_data = gene_filter.create_and_apply_mask(
            self.adata,
            self.sex_genes,
            self.autosomal_genes,
            self.autosomal_genes
        )
        
        # Should only have lncRNA genes
        lnc_genes = [name for name in filtered_data.var_names if name.startswith('lnc')]
        assert len(lnc_genes) == filtered_data.n_vars


class TestDataPreprocessor:
    """Test data preprocessing utilities."""
    
    def setUp(self):
        """Set up test data and configuration."""
        # Create mock AnnData objects
        n_obs, n_vars = 100, 1000
        X = np.random.randn(n_obs, n_vars)
        
        obs_data = pd.DataFrame({
            'sex': np.random.choice(['male', 'female', 'mix'], n_obs),
            'afca_annotation_broad': np.random.choice(['neuron', 'glia'], n_obs),
            'age': np.random.choice(['young', 'old'], n_obs)
        })
        
        var_names = [f"gene_{i}" for i in range(n_vars)]
        var_data = pd.DataFrame(index=var_names)
        
        self.adata = AnnData(X=X, obs=obs_data, var=var_data)
        self.adata_corrected = self.adata.copy()
        
        # Mock configuration
        self.mock_config = MagicMock()
        self.mock_config.DataParameters.Filtering.include_mixed_sex = True
        self.mock_config.DataParameters.GeneralSettings.sex_type = "all"
        self.mock_config.DataParameters.GeneralSettings.cell_type = "all"
        self.mock_config.GenePreprocessing.GeneShuffle.shuffle_genes = False
        self.mock_config.DataParameters.Sampling.num_samples = None
        self.mock_config.DataParameters.Sampling.num_variables = None
    
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        self.setUp()
        
        preprocessor = DataPreprocessor(
            self.mock_config,
            self.adata,
            self.adata_corrected
        )
        
        assert preprocessor.config == self.mock_config
        assert preprocessor.adata is self.adata
        assert preprocessor.adata_corrected is self.adata_corrected
    
    def test_process_adata_filter_sex(self):
        """Test filtering by sex type."""
        self.setUp()
        self.mock_config.DataParameters.GeneralSettings.sex_type = "male"
        
        preprocessor = DataPreprocessor(
            self.mock_config,
            self.adata,
            self.adata_corrected
        )
        
        processed = preprocessor.process_adata(self.adata)
        
        # Should only have male samples
        assert all(processed.obs['sex'] == 'male')
    
    def test_process_adata_filter_cell_type(self):
        """Test filtering by cell type."""
        self.setUp()
        self.mock_config.DataParameters.GeneralSettings.cell_type = "neuron"
        
        preprocessor = DataPreprocessor(
            self.mock_config,
            self.adata,
            self.adata_corrected
        )
        
        processed = preprocessor.process_adata(self.adata)
        
        # Should only have neuron samples
        assert all(processed.obs['afca_annotation_broad'] == 'neuron')
    
    def test_process_adata_sample_data(self):
        """Test data sampling."""
        self.setUp()
        self.mock_config.DataParameters.Sampling.num_samples = 50
        self.mock_config.DataSplit.random_state = 42
        
        preprocessor = DataPreprocessor(
            self.mock_config,
            self.adata,
            self.adata_corrected
        )
        
        processed = preprocessor.process_adata(self.adata)
        
        # Should have 50 samples
        assert processed.n_obs == 50


if __name__ == "__main__":
    pytest.main([__file__])