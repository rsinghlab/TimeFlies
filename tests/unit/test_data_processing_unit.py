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

    def test_data_preprocessor_initialization(self, small_sample_anndata):
        """Test DataPreprocessor can be initialized."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.project = "fruitfly_aging"

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)
        assert processor.config == config
        assert processor.adata is not None
        assert processor.adata_corrected is not None

    def test_process_adata(self, small_sample_anndata):
        """Test AnnData processing."""
        config = MagicMock()
        config.data.tissue = "head"
        config.data.normalize = True
        config.data.log_transform = True

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        # Test processing
        processed = processor.process_adata(small_sample_anndata.copy())

        assert processed is not None
        assert processed.n_obs <= small_sample_anndata.n_obs
        assert processed.n_vars <= small_sample_anndata.n_vars

    def test_split_data_random(self, small_sample_anndata):
        """Test random data splitting."""
        config = MagicMock()
        config.data.split.method = "random"
        config.data.split.test_size = 0.2
        config.general.random_state = 42

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)
        train, test = processor.split_data(small_sample_anndata)

        assert train.n_obs + test.n_obs == small_sample_anndata.n_obs
        assert train.n_vars == small_sample_anndata.n_vars
        assert test.n_vars == small_sample_anndata.n_vars

    def test_split_data_by_sex(self, small_sample_anndata):
        """Test data splitting by sex."""
        config = MagicMock()
        config.data.split.method = "sex"
        config.data.split.sex.train = "male"
        config.data.split.sex.test = "female"
        config.data.split.test_ratio = 0.5

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)
        train, test = processor.split_data(small_sample_anndata)

        assert train.n_obs > 0
        assert test.n_obs > 0

    def test_select_highly_variable_genes(self, small_sample_anndata):
        """Test highly variable gene selection."""
        config = MagicMock()
        config.gene_preprocessing.gene_filtering.n_top_genes = 50

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        # Mock scanpy highly variable genes
        with patch("scanpy.pp.highly_variable_genes"):
            with patch("scanpy.pp.filter_genes"):
                hvg_data = processor.select_highly_variable_genes(
                    small_sample_anndata.copy()
                )
                assert hvg_data is not None

    def test_prepare_labels(self, small_sample_anndata):
        """Test label preparation."""
        config = MagicMock()
        config.data.target_variable = "age"

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        labels, encoder = processor.prepare_labels(small_sample_anndata)

        assert labels is not None
        assert encoder is not None
        assert len(labels) == small_sample_anndata.n_obs

    def test_normalize_data(self, small_sample_anndata):
        """Test data normalization."""
        config = MagicMock()

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        # Mock scanpy normalization
        with patch("scanpy.pp.normalize_total"):
            with patch("scanpy.pp.log1p"):
                normalized = processor.normalize_data(small_sample_anndata.copy())
                assert normalized is not None

    def test_reshape_for_cnn(self, small_sample_anndata):
        """Test data reshaping for CNN."""
        config = MagicMock()

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        X = (
            small_sample_anndata.X.toarray()
            if hasattr(small_sample_anndata.X, "toarray")
            else small_sample_anndata.X
        )
        reshaped = processor.reshape_for_cnn(X)

        assert reshaped is not None
        assert len(reshaped.shape) == 2 or len(reshaped.shape) == 3

    def test_create_reference_data(self, small_sample_anndata):
        """Test reference data creation."""
        config = MagicMock()

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        X = (
            small_sample_anndata.X.toarray()
            if hasattr(small_sample_anndata.X, "toarray")
            else small_sample_anndata.X
        )
        reference = processor.create_reference_data(X)

        assert reference is not None
        assert reference.shape[1] == X.shape[1]

    def test_prepare_data(self, small_sample_anndata):
        """Test complete data preparation pipeline."""
        config = MagicMock()
        config.data.target_variable = "age"
        config.general.random_state = 42
        config.data.split.method = "random"
        config.data.split.test_size = 0.2
        config.model.type = "CNN"

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        # Mock the complex preparation pipeline
        with patch.object(
            processor, "process_adata", return_value=small_sample_anndata
        ):
            with patch.object(
                processor,
                "split_data",
                return_value=(small_sample_anndata[:8], small_sample_anndata[8:]),
            ):
                with patch.object(
                    processor,
                    "prepare_labels",
                    return_value=(np.array([0, 1, 2] * 3)[:8], MagicMock()),
                ):
                    result = processor.prepare_data()

                    assert result is not None
                    assert "X_train" in result
                    assert "X_test" in result
                    assert "y_train" in result
                    assert "y_test" in result

    def test_prepare_final_eval_data(self, small_sample_anndata):
        """Test final evaluation data preparation."""
        config = MagicMock()
        config.data.target_variable = "age"

        processor = DataPreprocessor(config, small_sample_anndata, small_sample_anndata)

        # Mock evaluation data preparation
        with patch.object(
            processor, "process_adata", return_value=small_sample_anndata
        ):
            with patch.object(
                processor,
                "prepare_labels",
                return_value=(np.array([0, 1, 2] * 4)[:10], MagicMock()),
            ):
                result = processor.prepare_final_eval_data(small_sample_anndata)

                assert result is not None
                assert "X_eval" in result
                assert "y_eval" in result
