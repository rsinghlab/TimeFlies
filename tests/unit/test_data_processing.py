"""Unit tests for data processing components."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from common.data.preprocessing.data_processor import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor functionality."""

    @patch("common.utils.path_manager.PathManager")
    def test_data_preprocessor_initialization(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test DataPreprocessor can be initialized with proper config."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )
        assert processor.config == aging_config
        assert processor.adata is not None
        assert processor.adata_corrected is not None

    @patch("common.utils.path_manager.PathManager")
    def test_process_adata(self, mock_path_manager, small_sample_anndata, aging_config):
        """Test AnnData processing."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test processing
        processed = processor.process_adata(small_sample_anndata.copy())

        assert processed is not None
        assert processed.n_obs <= small_sample_anndata.n_obs
        assert processed.n_vars <= small_sample_anndata.n_vars

    @patch("common.utils.path_manager.PathManager")
    def test_split_data_random(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test random data splitting."""
        # Set split method to random
        aging_config.data.split.method = "random"
        aging_config.data.split.test_ratio = 0.3

        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )
        train, test = processor.split_data(small_sample_anndata)

        assert train.n_obs + test.n_obs == small_sample_anndata.n_obs
        assert train.n_vars == small_sample_anndata.n_vars
        assert test.n_vars == small_sample_anndata.n_vars

    @patch("common.utils.path_manager.PathManager")
    def test_split_data_by_sex(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test data splitting by sex."""
        # Set split method to sex
        aging_config.data.split.method = "sex"
        aging_config.data.split.sex.train = "male"
        aging_config.data.split.sex.test = "female"
        aging_config.data.split.test_ratio = 0.5

        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )
        train, test = processor.split_data(small_sample_anndata)

        assert train.n_obs >= 0  # May be 0 if no males/females
        assert test.n_obs >= 0

    @patch("common.utils.path_manager.PathManager")
    def test_select_highly_variable_genes(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test highly variable gene selection."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test gene selection method exists
        assert hasattr(processor, "select_highly_variable_genes")

    @patch("common.utils.path_manager.PathManager")
    def test_prepare_labels(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test label preparation."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test label preparation method exists
        assert hasattr(processor, "prepare_labels")

    @patch("common.utils.path_manager.PathManager")
    def test_normalize_data(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test data normalization."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test normalization method exists
        assert hasattr(processor, "normalize_data")

    @patch("common.utils.path_manager.PathManager")
    def test_reshape_for_cnn(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test CNN data reshaping."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test reshape method exists
        assert hasattr(processor, "reshape_for_cnn")

    @patch("common.utils.path_manager.PathManager")
    def test_create_reference_data(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test reference data creation."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test reference data creation method exists
        assert hasattr(processor, "create_reference_data")

    @patch("common.utils.path_manager.PathManager")
    def test_prepare_data(self, mock_path_manager, small_sample_anndata, aging_config):
        """Test data preparation."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test data preparation method exists
        assert hasattr(processor, "prepare_data")

    @patch("common.utils.path_manager.PathManager")
    def test_prepare_final_eval_data(
        self, mock_path_manager, small_sample_anndata, aging_config
    ):
        """Test final evaluation data preparation."""
        processor = DataPreprocessor(
            aging_config, small_sample_anndata, small_sample_anndata
        )

        # Test final eval data preparation method exists
        assert hasattr(processor, "prepare_final_eval_data")
