"""Tests for path management functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from src.timeflies.utils.path_manager import PathManager


class TestPathManager:
    """Test PathManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test project structure
        self.project_root = Path(self.temp_dir) / "timeflies_test"
        self.project_root.mkdir()
        
        # Create expected directories
        (self.project_root / "data").mkdir()
        (self.project_root / "outputs").mkdir()
        (self.project_root / "src").mkdir()
        (self.project_root / "configs").mkdir()
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_mock_config(self, **overrides):
        """Create mock config with default values."""
        defaults = {
            "tissue": "head",
            "model_type": "cnn", 
            "encoding_variable": "age",
            "cell_type": "all",
            "sex_type": "all",
            "batch_correction_enabled": False
        }
        defaults.update(overrides)
        
        # Create nested mock structure
        config = Mock()
        config.data = Mock()
        config.data.tissue = defaults["tissue"]
        config.data.model_type = defaults["model_type"]
        config.data.encoding_variable = defaults["encoding_variable"]
        config.data.cell_type = defaults["cell_type"]
        config.data.sex_type = defaults["sex_type"]
        
        config.data.batch_correction = Mock()
        config.data.batch_correction.enabled = defaults["batch_correction_enabled"]
        
        # Gene preprocessing mocks
        config.gene_preprocessing = Mock()
        config.gene_preprocessing.gene_filtering = Mock()
        config.gene_preprocessing.gene_balancing = Mock()
        
        # Set default gene filtering options
        for attr in ["highly_variable_genes", "balance_genes", "balance_lnc_genes",
                    "select_batch_genes", "only_keep_lnc_genes", "remove_lnc_genes",
                    "remove_autosomal_genes", "remove_sex_genes"]:
            setattr(config.gene_preprocessing.gene_filtering, attr, False)
            setattr(config.gene_preprocessing.gene_balancing, attr, False)
            
        return config
        
    def test_path_manager_initialization(self):
        """Test PathManager initialization."""
        config = self.create_mock_config()
        
        # Mock the file location to point to our test directory
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            assert path_manager.tissue == "head"
            assert path_manager.model_type == "cnn"
            assert path_manager.encoding_variable == "age"
            assert path_manager.correction_dir == "uncorrected"
            
    def test_experiment_name_generation(self):
        """Test experiment name generation with naming convention."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            # Should generate: head_cnn_age/all-genes_all-cells_all-sexes
            assert path_manager.base_experiment == "head_cnn_age"
            assert path_manager.config_details == "all-genes_all-cells_all-sexes"
            assert path_manager.experiment_name == "head_cnn_age/all-genes_all-cells_all-sexes"
            
    def test_batch_correction_naming(self):
        """Test naming with batch correction enabled."""
        config = self.create_mock_config(batch_correction_enabled=True)
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            assert path_manager.correction_dir == "batch_corrected"
            
    def test_gene_method_detection(self):
        """Test gene method detection."""
        # Test HVG
        config = self.create_mock_config()
        config.gene_preprocessing.gene_filtering.highly_variable_genes = True
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            assert path_manager._get_gene_method() == "hvg"
            assert "hvg_all-cells_all-sexes" in path_manager.config_details
            
        # Test no-sex genes
        config = self.create_mock_config()
        config.gene_preprocessing.gene_filtering.remove_sex_genes = True
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            assert path_manager._get_gene_method() == "no-sex"
            
    def test_cell_type_naming(self):
        """Test cell type naming conventions."""
        config = self.create_mock_config(cell_type="muscle cell")
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            # Spaces should be converted to hyphens
            assert "muscle-cell" in path_manager.config_details
            
    def test_model_directory_construction(self):
        """Test model directory path construction."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            model_dir = path_manager.construct_model_directory()
            expected_path = self.project_root / "outputs" / "models" / "uncorrected" / "head_cnn_age" / "all-genes_all-cells_all-sexes"
            
            assert model_dir == str(expected_path)
            assert expected_path.exists()  # Should be created
            
    def test_processed_data_directory(self):
        """Test processed data directory construction."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            data_dir = path_manager.get_processed_data_dir()
            expected_path = self.project_root / "data" / "processed" / "uncorrected" / "head_cnn_age" / "all-genes_all-cells_all-sexes"
            
            assert data_dir == str(expected_path)
            assert expected_path.exists()
            
    def test_visualization_directory(self):
        """Test visualization directory construction."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            viz_dir = path_manager.get_visualization_directory()
            expected_path = self.project_root / "outputs" / "results" / "uncorrected" / "head_cnn_age" / "all-genes_all-cells_all-sexes"
            
            assert viz_dir == str(expected_path)
            assert expected_path.exists()
            
    def test_visualization_directory_with_subfolder(self):
        """Test visualization directory with subfolder."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            viz_dir = path_manager.get_visualization_directory("plots")
            expected_path = self.project_root / "outputs" / "results" / "uncorrected" / "head_cnn_age" / "all-genes_all-cells_all-sexes" / "plots"
            
            assert viz_dir == str(expected_path)
            assert expected_path.exists()
            
    def test_raw_data_directory(self):
        """Test raw data directory construction."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            raw_dir = path_manager.get_raw_data_dir()
            expected_path = self.project_root / "data" / "raw" / "h5ad" / "head" / "uncorrected"
            
            assert raw_dir == str(expected_path)
            
    def test_tissue_override(self):
        """Test tissue override in raw data directory."""
        config = self.create_mock_config(tissue="head")
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            raw_dir = path_manager.get_raw_data_dir(tissue_override="body")
            expected_path = self.project_root / "data" / "raw" / "h5ad" / "body" / "uncorrected"
            
            assert raw_dir == str(expected_path)
            
    def test_log_directory(self):
        """Test log directory construction."""
        config = self.create_mock_config()
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(Path, '__file__', str(self.project_root / "src" / "timeflies" / "utils" / "path_manager.py"))
            path_manager = PathManager(config)
            
            log_dir = path_manager.get_log_directory()
            expected_path = self.project_root / "outputs" / "logs"
            
            assert log_dir == str(expected_path)
            assert expected_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])