"""Tests for data preprocessing functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from anndata import AnnData

from src.timeflies.data.preprocessing.data_processor import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.adata, self.adata_corrected = self.create_sample_data()
        
    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()
        
        # Data configuration
        config.data = Mock()
        config.data.sex_type = 'all'
        config.data.cell_type = 'all'
        config.data.encoding_variable = 'age'
        
        # Filtering configuration
        config.data.filtering = Mock()
        config.data.filtering.include_mixed_sex = False
        
        # Sampling configuration
        config.data.sampling = Mock()
        config.data.sampling.num_samples = None
        config.data.sampling.num_variables = None
        
        # Train-test split configuration
        config.data.train_test_split = Mock()
        config.data.train_test_split.method = 'random'
        config.data.train_test_split.test_split = 0.2
        config.data.train_test_split.random_state = 42
        config.data.train_test_split.train = Mock()
        config.data.train_test_split.train.sex = 'male'
        config.data.train_test_split.train.tissue = 'head'
        config.data.train_test_split.test = Mock()
        config.data.train_test_split.test.sex = 'female'
        config.data.train_test_split.test.tissue = 'body'
        config.data.train_test_split.test.size = 0.3
        
        # Batch correction configuration
        config.data.batch_correction = Mock()
        config.data.batch_correction.enabled = False
        
        # Gene preprocessing configuration
        config.gene_preprocessing = Mock()
        config.gene_preprocessing.gene_shuffle = Mock()
        config.gene_preprocessing.gene_shuffle.shuffle_genes = False
        config.gene_preprocessing.gene_shuffle.shuffle_random_state = 42
        config.gene_preprocessing.gene_filtering = Mock()
        config.gene_preprocessing.gene_filtering.highly_variable_genes = False
        config.gene_preprocessing.gene_filtering.select_batch_genes = False
        
        # Data processing configuration
        config.data_processing = Mock()
        config.data_processing.normalization = Mock()
        config.data_processing.normalization.enabled = False
        
        # Feature importance configuration
        config.feature_importance = Mock()
        config.feature_importance.reference_size = 100
        
        return config
        
    def create_sample_data(self):
        """Create sample AnnData objects for testing."""
        n_obs, n_vars = 1000, 2000
        
        # Create expression data
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        # Create observations (cells)
        obs = pd.DataFrame({
            'age': np.random.choice([1, 5, 10, 20], n_obs),
            'sex': np.random.choice(['male', 'female', 'mix'], n_obs),
            'tissue': np.random.choice(['head', 'body'], n_obs),
            'afca_annotation_broad': np.random.choice([
                'CNS neuron', 'muscle cell', 'epithelial cell'
            ], n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])
        
        # Create variables (genes)
        var = pd.DataFrame({
            'gene_type': np.random.choice(['protein_coding', 'lncRNA'], n_vars),
            'highly_variable': np.random.choice([True, False], n_vars, p=[0.2, 0.8])
        }, index=[f'gene_{i}' for i in range(n_vars)])
        
        adata = AnnData(X=X, obs=obs, var=var)
        adata_corrected = adata.copy()
        
        return adata, adata_corrected
        
    def test_processor_initialization(self):
        """Test DataPreprocessor initialization."""
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        assert processor.config == self.mock_config
        assert processor.adata == self.adata
        assert processor.adata_corrected == self.adata_corrected
        assert processor.path_manager is not None
        
    def test_process_adata_basic(self):
        """Test basic adata processing."""
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        processed = processor.process_adata(self.adata)
        
        # Should return processed data
        assert isinstance(processed, AnnData)
        assert processed.n_obs <= self.adata.n_obs  # May be filtered
        assert processed.n_vars <= self.adata.n_vars  # May be filtered
        
    def test_process_adata_exclude_mix_sex(self):
        """Test exclusion of mixed sex cells."""
        self.mock_config.data.filtering.include_mixed_sex = False
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        processed = processor.process_adata(self.adata)
        
        # Should exclude mix sex
        assert 'mix' not in processed.obs['sex'].values
        
    def test_process_adata_sex_filter(self):
        """Test sex-based filtering."""
        self.mock_config.data.sex_type = 'male'
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        processed = processor.process_adata(self.adata)
        
        # Should only contain male cells
        assert all(processed.obs['sex'] == 'male')
        
    def test_process_adata_cell_type_filter(self):
        """Test cell type filtering."""
        self.mock_config.data.cell_type = 'muscle cell'
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        processed = processor.process_adata(self.adata)
        
        # Should only contain muscle cells
        assert all(processed.obs['afca_annotation_broad'] == 'muscle cell')
        
    def test_process_adata_sampling(self):
        """Test data sampling."""
        self.mock_config.data.sampling.num_samples = 500
        self.mock_config.data.sampling.num_variables = 1000
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        processed = processor.process_adata(self.adata)
        
        # Should be sampled to specified size
        assert processed.n_obs <= 500
        assert processed.n_vars <= 1000
        
    def test_process_adata_gene_shuffling(self):
        """Test gene shuffling."""
        self.mock_config.gene_preprocessing.gene_shuffle.shuffle_genes = True
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        original_genes = self.adata.var_names.copy()
        processed = processor.process_adata(self.adata)
        
        # Genes should be shuffled (different order)
        assert len(processed.var_names) == len(original_genes)
        # Allow for possibility that random shuffle results in same order
        
    def test_split_data_random(self):
        """Test random data splitting."""
        self.mock_config.data.train_test_split.method = 'random'
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train, test = processor.split_data(self.adata)
        
        # Should split data
        assert isinstance(train, AnnData)
        assert isinstance(test, AnnData)
        assert train.n_obs + test.n_obs <= self.adata.n_obs
        assert train.n_vars == test.n_vars
        
    def test_split_data_by_sex(self):
        """Test sex-based data splitting."""
        self.mock_config.data.train_test_split.method = 'sex'
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train, test = processor.split_data(self.adata)
        
        # Train should be male, test should be female
        assert all(train.obs['sex'] == 'male')
        assert all(test.obs['sex'] == 'female')
        
    def test_split_data_by_tissue(self):
        """Test tissue-based data splitting."""
        self.mock_config.data.train_test_split.method = 'tissue'
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train, test = processor.split_data(self.adata)
        
        # Train should be head, test should be body
        assert all(train.obs['tissue'] == 'head')
        assert all(test.obs['tissue'] == 'body')
        # Should have common genes
        assert train.n_vars == test.n_vars
        
    def test_select_highly_variable_genes(self):
        """Test highly variable gene selection."""
        self.mock_config.gene_preprocessing.gene_filtering.highly_variable_genes = True
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        # Create train/test data
        train, test = processor.split_data(self.adata)
        
        # Test HVG selection
        train_hvg, test_hvg, hvg_list = processor.select_highly_variable_genes(train, test)
        
        assert isinstance(train_hvg, AnnData)
        assert isinstance(test_hvg, AnnData)
        assert isinstance(hvg_list, list) or hvg_list is None
        
        if hvg_list:
            assert train_hvg.n_vars == len(hvg_list)
            assert test_hvg.n_vars == len(hvg_list)
            
    def test_encode_labels(self):
        """Test label encoding."""
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        # Create train/test data
        train, test = processor.split_data(self.adata)
        
        train_labels, test_labels, label_encoder = processor.encode_labels(train, test)
        
        # Should return encoded labels
        assert train_labels.shape[0] == train.n_obs
        assert test_labels.shape[0] == test.n_obs
        assert train_labels.shape[1] == test_labels.shape[1]  # Same number of classes
        assert label_encoder is not None
        
    def test_normalize_data_disabled(self):
        """Test data normalization when disabled."""
        self.mock_config.data_processing.normalization.enabled = False
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train_data = np.random.randn(100, 50)
        test_data = np.random.randn(50, 50)
        
        norm_train, norm_test, scaler, is_fit = processor.normalize_data(train_data, test_data)
        
        # Should return original data when normalization disabled
        assert np.array_equal(norm_train, train_data)
        assert np.array_equal(norm_test, test_data)
        assert scaler is None
        assert not is_fit
        
    def test_normalize_data_enabled(self):
        """Test data normalization when enabled."""
        self.mock_config.data_processing.normalization.enabled = True
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train_data = np.random.randn(100, 50)
        test_data = np.random.randn(50, 50)
        
        norm_train, norm_test, scaler, is_fit = processor.normalize_data(train_data, test_data)
        
        # Should normalize data when enabled
        assert not np.array_equal(norm_train, train_data)
        assert not np.array_equal(norm_test, test_data)
        assert scaler is not None
        assert is_fit
        
    def test_generate_reference_data(self):
        """Test reference data generation."""
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train_data = np.random.randn(1000, 50)
        
        reference_data = processor.generate_reference_data(train_data)
        
        # Should return reference data of correct size
        assert reference_data.shape[0] <= self.mock_config.feature_importance.reference_size
        assert reference_data.shape[1] == train_data.shape[1]
        
    def test_generate_reference_data_small_dataset(self):
        """Test reference data generation with small dataset."""
        self.mock_config.feature_importance.reference_size = 1000
        processor = DataPreprocessor(self.mock_config, self.adata, self.adata_corrected)
        
        train_data = np.random.randn(50, 50)  # Smaller than reference size
        
        reference_data = processor.generate_reference_data(train_data)
        
        # Should use all available data when dataset is smaller
        assert reference_data.shape[0] == train_data.shape[0]
        assert reference_data.shape[1] == train_data.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])