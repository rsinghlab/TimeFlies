"""Integration tests for DATA PIPELINE components.

Tests how data-related components work together:
- DataLoader + PathManager + DataSetupManager  
- Configuration loading + data file discovery
- Data preprocessing pipelines with real AnnData objects
- File I/O and directory structure handling

Focus: Data flow from raw files through preprocessing
Time: ~30-60 seconds with small real data
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch
import scanpy as sc
from anndata import AnnData

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.core.config_manager import ConfigManager, Config
from shared.data.loaders import DataLoader
from shared.data.setup import DataSetupManager
from shared.utils.path_manager import PathManager


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create project structure
        project_dir = Path(temp_dir)
        data_dir = project_dir / "data" / "fruitfly_aging" / "head"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create the old structure that PathManager looks for in test environments
        old_data_dir = project_dir / "data" / "raw" / "h5ad"
        old_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create src directory so PathManager recognizes this as project root
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "configs").mkdir(exist_ok=True)
        (project_dir / "outputs").mkdir(exist_ok=True)
        
        # Create sample AnnData objects for testing
        n_obs, n_vars = 1000, 500
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        obs = pd.DataFrame({
            'age': np.random.choice([5, 30, 50, 70], n_obs),
            'sex': np.random.choice(['male', 'female'], n_obs),
            'tissue': 'head',
            'batch': np.random.choice(['batch1', 'batch2'], n_obs),
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        var = pd.DataFrame({
            'gene_symbol': [f'gene_{i}' for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], n_vars, p=[0.2, 0.8])
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        # Create test data files
        adata_original = AnnData(X=X, obs=obs, var=var)
        adata_original.write(data_dir / "drosophila_head_aging_original.h5ad")
        
        # Create train/eval splits
        train_indices = np.random.choice(n_obs, size=int(0.8 * n_obs), replace=False)
        eval_indices = np.setdiff1d(np.arange(n_obs), train_indices)
        
        adata_train = adata_original[train_indices].copy()
        adata_eval = adata_original[eval_indices].copy()
        
        adata_train.write(data_dir / "drosophila_head_aging_train.h5ad")
        adata_eval.write(data_dir / "drosophila_head_aging_eval.h5ad")
        
        # Create gene lists
        autosomal_genes = [f'gene_{i}' for i in range(400)]  # Most genes autosomal
        sex_genes = [f'gene_{i}' for i in range(400, 500)]    # Last 100 genes sex-linked
        
        pd.DataFrame(autosomal_genes).to_csv(data_dir / "autosomal_genes.csv", header=False, index=False)
        pd.DataFrame(sex_genes).to_csv(data_dir / "sex_genes.csv", header=False, index=False)
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def real_config(self, temp_project_dir):
        """Create a real configuration for testing."""
        config_dict = {
            'general': {
                'project_name': 'TimeFlies',
                'version': '0.2.0',
                'random_state': 42
            },
            'data': {
                'tissue': 'head',
                'species': 'drosophila',
                'project': 'fruitfly_aging',
                'model_type': 'CNN',
                'encoding_variable': 'age',
                'cell_type': 'all',
                'sex_type': 'all',
                'batch_correction': {'enabled': False},
                'filtering': {'include_mixed_sex': False},
                'sampling': {'num_samples': None, 'num_variables': None},
                'train_test_split': {
                    'test_split': 0.2,
                    'random_state': 42
                }
            },
            'gene_preprocessing': {
                'gene_filtering': {
                    'remove_sex_genes': False,
                    'remove_autosomal_genes': False,
                    'highly_variable_genes': False,
                    'select_batch_genes': False
                },
                'gene_balancing': {
                    'balance_genes': False
                }
            },
            'file_locations': {
                'training_file': "drosophila_head_aging_train.h5ad",
                'evaluation_file': "drosophila_head_aging_eval.h5ad", 
                'original_file': "drosophila_head_aging_original.h5ad",
                'batch_corrected_files': {
                    'train': "drosophila_head_aging_train_batch.h5ad",
                    'eval': "drosophila_head_aging_eval_batch.h5ad"
                }
            }
        }
        
        return Config(config_dict)
    
    def test_path_manager_real_paths(self, temp_project_dir, real_config):
        """Test PathManager with real directory structure."""
        # Change directory to temp project
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            path_manager = PathManager(real_config)
            
            # Test raw data directory
            raw_data_dir = path_manager.get_raw_data_dir()
            expected_path = temp_project_dir / "data" / "fruitfly_aging" / "head"
            assert raw_data_dir == str(expected_path)
            
            # Test that the directory actually exists
            assert Path(raw_data_dir).exists()
            
            # Test tissue override
            body_dir = path_manager.get_raw_data_dir(tissue_override="body")
            expected_body_path = temp_project_dir / "data" / "fruitfly_aging" / "body"
            assert body_dir == str(expected_body_path)
        finally:
            os.chdir(original_cwd)
    
    def test_data_loader_real_files(self, temp_project_dir, real_config):
        """Test DataLoader with real files."""
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            loader = DataLoader(real_config)
            
            # Test that files are found correctly
            expected_data_dir = temp_project_dir / "data" / "fruitfly_aging" / "head"
            assert Path(loader.Data_dir) == expected_data_dir
            
            # Test loading actual data
            adata, adata_eval, adata_original = loader.load_data()
            
            # Verify data was loaded correctly
            assert isinstance(adata, AnnData)
            assert isinstance(adata_eval, AnnData)  
            assert isinstance(adata_original, AnnData)
            
            # Check data dimensions make sense
            assert adata.n_obs > 0
            assert adata.n_vars == 500  # From our test data
            assert adata_eval.n_obs > 0
            assert adata_original.n_obs == 1000  # From our test data
            
            # Test gene list loading
            autosomal_genes, sex_genes = loader.load_gene_lists()
            assert len(autosomal_genes) == 400  # From our test data
            assert len(sex_genes) == 100       # From our test data
            assert all(isinstance(gene, str) for gene in autosomal_genes)
            assert all(isinstance(gene, str) for gene in sex_genes)
        finally:
            os.chdir(original_cwd)
    
    def test_config_manager_real_functionality(self):
        """Test ConfigManager with real configuration operations."""
        # Test creating config from dictionary
        config_dict = {
            'general': {'project_name': 'TestProject'},
            'data': {'tissue': 'head', 'model_type': 'CNN'},
            'nested': {'level2': {'level3': 'value'}}
        }
        
        config = ConfigManager.from_dict(config_dict)
        
        # Test nested access works
        assert config.general.project_name == 'TestProject'
        assert config.data.tissue == 'head'
        assert config.nested.level2.level3 == 'value'
        
        # Test config modification
        config.data.tissue = 'body'
        assert config.data.tissue == 'body'
        
        # Test get method with defaults
        assert config.get('nonexistent') is None
        assert config.get('nonexistent', 'default') == 'default'
        
        # Test to_dict conversion
        result_dict = config.to_dict()
        assert result_dict['general']['project_name'] == 'TestProject'
        assert result_dict['data']['tissue'] == 'body'  # Modified value
    
    def test_data_processing_pipeline_integration(self, temp_project_dir, real_config):
        """Test the complete data processing pipeline."""
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            # Test DataLoader
            loader = DataLoader(real_config)
            adata, adata_eval, adata_original = loader.load_data()
            
            # Verify we have actual biological data structure
            assert 'age' in adata.obs.columns
            assert 'sex' in adata.obs.columns
            assert 'tissue' in adata.obs.columns
            
            # Test age groups are reasonable
            unique_ages = adata.obs['age'].unique()
            assert len(unique_ages) > 1  # Should have multiple age groups
            assert all(age in [5, 30, 50, 70] for age in unique_ages)  # From our test data
            
            # Test gene expression data
            assert adata.X.shape[1] == 500  # Number of genes
            assert np.all(adata.X >= 0)     # Gene expression should be non-negative
            
            # Test that train and eval splits are different
            assert not np.array_equal(adata.X, adata_eval.X)
            assert adata.n_obs != adata_eval.n_obs
            
            # Test that original contains both train and eval
            assert adata_original.n_obs >= adata.n_obs + adata_eval.n_obs
        finally:
            os.chdir(original_cwd)
    
    def test_data_setup_manager_functionality(self, temp_project_dir, real_config):
        """Test DataSetupManager with real file operations."""
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            setup_manager = DataSetupManager(real_config)
            
            # Test filename generation
            train_filename = setup_manager._generate_filename('head', 'train')
            assert train_filename == 'drosophila_head_aging_train.h5ad'
            
            batch_filename = setup_manager._generate_filename('head', 'train', batch=True)
            assert batch_filename == 'drosophila_head_aging_train_batch.h5ad'
            
            # Test basic functionality (skip methods that don't exist)
            # Just verify setup manager was created successfully
            assert setup_manager is not None
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling_real_scenarios(self, temp_project_dir, real_config):
        """Test error handling with real file system scenarios."""
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            # Test with missing files
            bad_config = Config({
                'general': real_config.general.to_dict(),
                'data': {**real_config.data.to_dict(), 'tissue': 'nonexistent'},
                'gene_preprocessing': real_config.gene_preprocessing.to_dict(),
                'file_locations': real_config.file_locations.to_dict()
            })
            
            loader = DataLoader(bad_config)
            
            # Should handle missing directory gracefully
            with pytest.raises((FileNotFoundError, OSError)):
                loader.load_data()
        finally:
            os.chdir(original_cwd)
    
    def test_cross_tissue_functionality(self, temp_project_dir, real_config):
        """Test cross-tissue analysis capabilities.""" 
        # Create body tissue data
        body_dir = temp_project_dir / "data" / "fruitfly_aging" / "body"
        body_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy head data to body for testing
        head_dir = temp_project_dir / "data" / "fruitfly_aging" / "head"
        for file in head_dir.glob("*.h5ad"):
            shutil.copy(file, body_dir / file.name.replace('head', 'body'))
        
        original_cwd = os.getcwd()
        os.chdir(temp_project_dir)
        
        try:
            path_manager = PathManager(real_config)
            
            # Test accessing different tissues
            head_dir = path_manager.get_raw_data_dir(tissue_override="head")
            body_dir = path_manager.get_raw_data_dir(tissue_override="body")
            
            assert 'head' in head_dir
            assert 'body' in body_dir
            assert head_dir != body_dir
            
            # Both directories should exist
            assert Path(head_dir).exists()
            assert Path(body_dir).exists()
        finally:
            os.chdir(original_cwd)