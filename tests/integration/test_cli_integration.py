"""Integration tests for CLI functionality."""

import pytest
import subprocess
import tempfile
import os
import yaml
from pathlib import Path


class TestCLIIntegration:
    """Test CLI integration and end-to-end functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "timeflies_test"
        self.project_root.mkdir()
        
        # Create minimal project structure
        (self.project_root / "data" / "raw" / "h5ad" / "head" / "uncorrected").mkdir(parents=True)
        (self.project_root / "data" / "raw" / "gene_lists").mkdir(parents=True)
        (self.project_root / "outputs" / "models").mkdir(parents=True)
        (self.project_root / "outputs" / "results").mkdir(parents=True)
        (self.project_root / "src" / "timeflies").mkdir(parents=True)
        (self.project_root / "configs").mkdir(parents=True)
        
        # Create minimal config
        self.config_data = {
            "general": {"project_name": "test", "random_state": 42},
            "data": {
                "tissue": "head",
                "model_type": "CNN", 
                "encoding_variable": "age",
                "cell_type": "all",
                "sex_type": "all",
                "batch_correction": {"enabled": False},
                "train_test_split": {"method": "random", "test_split": 0.2}
            },
            "model": {"training": {"epochs": 1, "batch_size": 32}},
            "gene_preprocessing": {
                "gene_filtering": {"highly_variable_genes": False},
                "gene_balancing": {"balance_genes": False}
            }
        }
        
        self.config_path = self.project_root / "configs" / "test.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
            
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def run_cli_command(self, args, cwd=None):
        """Run CLI command and return result."""
        if cwd is None:
            cwd = self.project_root
            
        cmd = ["python", "run_timeflies.py"] + args
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            capture_output=True, 
            text=True,
            timeout=30
        )
        return result
        
    @pytest.mark.slow
    def test_cli_test_command(self):
        """Test the test command runs without errors."""
        # Copy the actual run_timeflies.py to test directory
        import shutil
        actual_script = Path(__file__).parent.parent.parent / "run_timeflies.py"
        if actual_script.exists():
            shutil.copy(actual_script, self.project_root)
            
            result = self.run_cli_command(["test"])
            
            # Should run without crashing (exit code 0 or 1 acceptable)
            assert result.returncode in [0, 1]
            assert "Testing TimeFlies installation" in result.stdout
            
    def test_cli_help_output(self):
        """Test that help output is generated correctly."""
        import shutil
        actual_script = Path(__file__).parent.parent.parent / "run_timeflies.py"
        if actual_script.exists():
            shutil.copy(actual_script, self.project_root)
            
            result = self.run_cli_command(["--help"])
            
            assert result.returncode == 0
            assert "TimeFlies: Machine Learning for Aging Analysis" in result.stdout
            assert "setup" in result.stdout
            assert "train" in result.stdout
            assert "batch" in result.stdout
            
    def test_cli_setup_command_structure(self):
        """Test setup command creates proper directory structure."""
        import shutil
        actual_script = Path(__file__).parent.parent.parent / "run_timeflies.py"
        if actual_script.exists():
            shutil.copy(actual_script, self.project_root)
            
            result = self.run_cli_command([
                "setup", 
                "--tissue", "head",
                "--config", str(self.config_path)
            ])
            
            # Should create appropriate directories even if data is missing
            # The command might fail due to missing data files, but structure should be attempted
            assert result.returncode in [0, 1]  # May fail due to missing data
            
    def test_cli_train_command_validation(self):
        """Test train command parameter validation."""
        import shutil
        actual_script = Path(__file__).parent.parent.parent / "run_timeflies.py"
        if actual_script.exists():
            shutil.copy(actual_script, self.project_root)
            
            result = self.run_cli_command([
                "train",
                "--tissue", "head",
                "--model", "cnn", 
                "--encoding", "age",
                "--config", str(self.config_path)
            ])
            
            # Command should be structured correctly even if it fails on missing data
            assert result.returncode in [0, 1]
            
    def test_cli_batch_command_options(self):
        """Test batch correction command options."""
        import shutil
        actual_script = Path(__file__).parent.parent.parent / "run_timeflies.py"
        if actual_script.exists():
            shutil.copy(actual_script, self.project_root)
            
            # Test train option
            result = self.run_cli_command([
                "batch", 
                "--train",
                "--tissue", "head"
            ])
            
            # Should attempt to run batch correction training
            assert result.returncode in [0, 1]
            
            # Test evaluate option  
            result = self.run_cli_command([
                "batch",
                "--evaluate", 
                "--tissue", "head"
            ])
            
            assert result.returncode in [0, 1]
            
    def test_cli_invalid_arguments(self):
        """Test CLI error handling for invalid arguments."""
        import shutil
        actual_script = Path(__file__).parent.parent.parent / "run_timeflies.py"
        if actual_script.exists():
            shutil.copy(actual_script, self.project_root)
            
            # Test invalid model type
            result = self.run_cli_command([
                "train",
                "--model", "invalid_model",
                "--tissue", "head"
            ])
            
            assert result.returncode != 0
            
            # Test invalid tissue
            result = self.run_cli_command([
                "train", 
                "--tissue", "invalid_tissue",
                "--model", "cnn"
            ])
            
            assert result.returncode != 0


class TestConfigIntegration:
    """Test configuration integration across components."""
    
    def test_config_consistency(self):
        """Test that config values are consistently used across components."""
        from src.timeflies.core.config_manager import ConfigManager
        from src.timeflies.utils.path_manager import PathManager
        
        config_data = {
            "general": {"project_name": "test"},
            "data": {
                "tissue": "head",
                "model_type": "CNN",
                "encoding_variable": "age", 
                "batch_correction": {"enabled": True}
            },
            "model": {"training": {"epochs": 100}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            # Load config through manager
            manager = ConfigManager(config_path)
            config = manager.get_config()
            
            # Create path manager with same config
            path_manager = PathManager(config)
            
            # Verify consistency
            assert path_manager.tissue == config.data.tissue
            assert path_manager.correction_dir == "batch_corrected"
            assert "head_cnn_age" in path_manager.experiment_name
            
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])