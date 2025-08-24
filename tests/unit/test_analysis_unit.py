"""Tests for analysis modules (EDA and visualization functionality)."""

# Import from project-specific modules since analysis is project-specific
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the module first to ensure it's loaded
from common.analysis.eda import EDAHandler
from common.analysis.visuals import VisualizationTools as Visualizer


class TestEDAHandler:
    """Test EDAHandler functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.mock_config = self.create_mock_config()
        self.mock_path_manager = Mock()
        self.adata, self.adata_eval, self.adata_original = self.create_sample_data()
        self.adata_corrected, self.adata_eval_corrected = self.create_corrected_data()

    def create_mock_config(self):
        """Create mock configuration object."""
        config = Mock()

        # Data configuration
        config.data = Mock()
        config.data.target_variable = 'age'

        # Batch correction configuration
        config.data.batch_correction = Mock()
        config.data.batch_correction.enabled = False

        return config

    def create_sample_data(self):
        """Create sample AnnData objects for testing."""
        n_obs, n_vars = 1000, 2000

        # Create expression data
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)

        # Create observations (cells)
        obs = pd.DataFrame({
            'age': np.random.choice([1, 5, 10, 20], n_obs),
            'sex': np.random.choice(['male', 'female'], n_obs),
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
        adata_eval = adata[::2].copy()  # Every other cell for eval
        adata_original = adata.copy()

        return adata, adata_eval, adata_original

    def create_corrected_data(self):
        """Create batch-corrected sample data."""
        adata_corrected = self.adata.copy()
        adata_eval_corrected = self.adata_eval.copy()

        # Add some variation to simulate batch correction
        adata_corrected.X = adata_corrected.X + np.random.normal(0, 0.1, adata_corrected.X.shape)
        adata_eval_corrected.X = adata_eval_corrected.X + np.random.normal(0, 0.1, adata_eval_corrected.X.shape)

        return adata_corrected, adata_eval_corrected

    @patch('common.analysis.eda.VisualizationTools')
    def test_eda_handler_initialization(self, mock_vis_tools):
        """Test EDAHandler initialization."""
        handler = EDAHandler(
            self.mock_config, self.mock_path_manager,
            self.adata, self.adata_eval, self.adata_original,
            self.adata_corrected, self.adata_eval_corrected
        )

        assert handler.config == self.mock_config
        assert handler.path_manager == self.mock_path_manager
        assert handler.adata is self.adata
        assert handler.adata_eval is self.adata_eval
        assert handler.adata_original is self.adata_original
        assert handler.adata_corrected is self.adata_corrected
        assert handler.adata_eval_corrected is self.adata_eval_corrected

        # Check that VisualizationTools is initialized
        mock_vis_tools.assert_called_once_with(
            config=self.mock_config, path_manager=self.mock_path_manager
        )

    @patch('common.analysis.eda.VisualizationTools')
    def test_run_eda_uncorrected_data(self, mock_vis_tools):
        """Test running EDA on uncorrected data."""
        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools

        handler = EDAHandler(
            self.mock_config, self.mock_path_manager,
            self.adata, self.adata_eval, self.adata_original,
            self.adata_corrected, self.adata_eval_corrected
        )

        with patch.object(handler, 'eda') as mock_eda:
            handler.run_eda()

            # Should call eda 3 times for uncorrected data (train, eval, original)
            assert mock_eda.call_count == 3

            # Check the calls
            calls = mock_eda.call_args_list

            # First call: training data
            assert calls[0][1]['dataset_name'] == 'train'
            assert calls[0][1]['folder_name'] == 'Training Data'
            assert calls[0][1]['encoding_column'] == 'age'

            # Second call: evaluation data
            assert calls[1][1]['dataset_name'] == 'evaluation'
            assert calls[1][1]['folder_name'] == 'Evaluation Data'

            # Third call: original data
            assert calls[2][1]['dataset_name'] == 'original'
            assert calls[2][1]['folder_name'] == 'Original Data'

    @patch('common.analysis.eda.VisualizationTools')
    def test_run_eda_batch_corrected_data(self, mock_vis_tools):
        """Test running EDA on batch-corrected data."""
        # Enable batch correction
        self.mock_config.data.batch_correction.enabled = True

        mock_visual_tools = Mock()
        mock_vis_tools.return_value = mock_visual_tools

        handler = EDAHandler(
            self.mock_config, self.mock_path_manager,
            self.adata, self.adata_eval, self.adata_original,
            self.adata_corrected, self.adata_eval_corrected
        )

        with patch.object(handler, 'eda') as mock_eda:
            handler.run_eda()

            # Should call eda 2 times for batch-corrected data (train, eval)
            assert mock_eda.call_count == 2

            # Check the calls
            calls = mock_eda.call_args_list

            # First call: batch training data
            assert calls[0][1]['dataset_name'] == 'batch_train'
            assert calls[0][1]['folder_name'] == 'Batch Training Data'

            # Second call: batch evaluation data
            assert calls[1][1]['dataset_name'] == 'batch_evaluation'
            assert calls[1][1]['folder_name'] == 'Batch Evaluation Data'




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
