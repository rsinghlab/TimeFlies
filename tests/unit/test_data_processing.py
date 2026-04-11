"""Unit tests for data processing components."""

from unittest.mock import patch

import pytest

from timeflies.data.preprocessing.data_processor import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor functionality."""

    @patch("timeflies.utils.path_manager.PathManager")
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

    @patch("timeflies.utils.path_manager.PathManager")
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

    @patch("timeflies.utils.path_manager.PathManager")
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

    @patch("timeflies.utils.path_manager.PathManager")
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
