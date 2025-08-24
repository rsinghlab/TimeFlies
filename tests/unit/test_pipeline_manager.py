"""Tests for PipelineManager - updated for current API."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestPipelineManager:
    """Test PipelineManager functionality with current API."""

    def test_pipeline_manager_initialization_mocked(self):
        """Test PipelineManager initialization with mocked dependencies."""
        from unittest.mock import patch

        # Mock all the dependencies to avoid complex initialization
        with patch("common.core.pipeline_manager.DataLoader") as mock_data_loader:
            with patch("common.core.pipeline_manager.PathManager") as mock_path_manager:
                with patch(
                    "common.core.pipeline_manager.StorageManager"
                ) as mock_storage_manager:
                    # Create a mock config
                    mock_config = MagicMock()

                    # Import and create PipelineManager
                    from common.core.pipeline_manager import PipelineManager

                    pipeline = PipelineManager(mock_config)

                    # Verify initialization
                    assert pipeline.config_instance == mock_config
                    mock_data_loader.assert_called_once_with(mock_config)
                    mock_path_manager.assert_called_once_with(mock_config)
                    mock_storage_manager.assert_called_once()

    def test_load_data_method_exists(self):
        """Test that load_data method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Test method exists without calling it
                    assert hasattr(pipeline, "load_data")
                    assert callable(getattr(pipeline, "load_data"))

    def test_run_method_exists(self):
        """Test that run method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Test method exists without calling it
                    assert hasattr(pipeline, "run")
                    assert callable(getattr(pipeline, "run"))

    def test_run_evaluation_method_exists(self):
        """Test that run_evaluation method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Test method exists without calling it
                    assert hasattr(pipeline, "run_evaluation")
                    assert callable(getattr(pipeline, "run_evaluation"))

    def test_load_or_train_model_method_exists(self):
        """Test that load_or_train_model method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Mock dependencies
                    pipeline.load_model = MagicMock()
                    pipeline.build_and_train_model = MagicMock()

                    # Test method exists and can be called
                    pipeline.load_or_train_model()

                    # At least one of the methods should be attempted
                    assert (
                        pipeline.load_model.called
                        or pipeline.build_and_train_model.called
                    )

    def test_preprocess_data_method_exists(self):
        """Test that preprocess_data method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Mock dependencies to avoid actual data processing
                    pipeline.data_loader.load_data = MagicMock()

                    # Test method exists (actual preprocessing would require real data)
                    assert hasattr(pipeline, "preprocess_data")
                    assert callable(getattr(pipeline, "preprocess_data"))

    def test_run_interpretation_method_exists(self):
        """Test that run_interpretation method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Test method exists
                    assert hasattr(pipeline, "run_interpretation")
                    assert callable(getattr(pipeline, "run_interpretation"))

    def test_run_visualizations_method_exists(self):
        """Test that run_visualizations method exists."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    pipeline = PipelineManager(mock_config)

                    # Test method exists
                    assert hasattr(pipeline, "run_visualizations")
                    assert callable(getattr(pipeline, "run_visualizations"))


@pytest.mark.unit
class TestPipelineManagerMethods:
    """Test individual PipelineManager methods with proper mocking."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a properly mocked PipelineManager."""
        from unittest.mock import patch

        with patch("common.core.pipeline_manager.DataLoader"):
            with patch("common.core.pipeline_manager.PathManager"):
                with patch("common.core.pipeline_manager.StorageManager"):
                    from common.core.pipeline_manager import PipelineManager

                    mock_config = MagicMock()
                    return PipelineManager(mock_config)

    def test_setup_gpu_method(self, mock_pipeline):
        """Test GPU setup method."""
        # Mock GPU handler
        with patch("common.core.pipeline_manager.GPUHandler"):
            mock_pipeline.setup_gpu()
            # Method should complete without error
            assert True

    def test_display_duration_method(self, mock_pipeline):
        """Test duration display method."""
        import time

        start_time = time.time()
        end_time = start_time + 5

        # Test method exists and doesn't crash
        mock_pipeline.display_duration(start_time, end_time)
        assert True
