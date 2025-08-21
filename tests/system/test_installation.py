"""System tests for TimeFlies installation and environment validation.

These tests verify that TimeFlies is properly installed and all dependencies
are available. Similar to `python run_timeflies.py test` but as pytest.
"""

import pytest
import importlib
import sys
from pathlib import Path


class TestInstallation:
    """Test TimeFlies installation and dependencies."""
    
    def test_core_imports(self):
        """Test that core TimeFlies modules can be imported."""
        # Add src to path for imports
        src_path = Path(__file__).parent.parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Test core imports
        from shared.core.config_manager import ConfigManager, Config
        from shared.core.pipeline_manager import PipelineManager
        from shared.utils.path_manager import PathManager
        
        # Verify they can be instantiated
        assert ConfigManager is not None
        assert Config is not None  
        assert PipelineManager is not None
        assert PathManager is not None
    
    def test_machine_learning_dependencies(self):
        """Test that ML dependencies are available."""
        import tensorflow as tf
        import sklearn
        import numpy as np
        import pandas as pd
        
        # Verify versions are reasonable
        assert hasattr(tf, '__version__')
        assert hasattr(sklearn, '__version__')
        assert hasattr(np, '__version__')
        assert hasattr(pd, '__version__')
    
    def test_single_cell_dependencies(self):
        """Test that single-cell analysis dependencies are available.""" 
        import scanpy as sc
        import anndata as ad
        
        assert hasattr(sc, '__version__')
        assert hasattr(ad, '__version__')
    
    def test_visualization_dependencies(self):
        """Test that visualization dependencies are available."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        assert plt is not None
        assert sns is not None
    
    def test_interpretation_dependencies(self):
        """Test that model interpretation dependencies are available."""
        import shap
        
        assert hasattr(shap, '__version__')
    
    def test_optional_batch_correction_dependencies(self):
        """Test batch correction dependencies (optional - may not be installed)."""
        try:
            import scvi
            import torch
            print("✅ Batch correction packages available")
        except ImportError:
            pytest.skip("Batch correction packages not installed (optional)")
    
    def test_configuration_loading(self):
        """Test that configuration files can be loaded."""
        from shared.core.config_manager import ConfigManager
        
        # Test config creation with existing config file
        config_manager = ConfigManager('configs/fruitfly_aging/default.yaml')
        config = config_manager.get_config()
        
        assert hasattr(config, 'data')
        assert hasattr(config, 'general')
        assert hasattr(config.data, 'tissue')
        assert config.data.tissue == 'head'
    
    def test_project_structure(self):
        """Test that expected project directories exist."""
        project_root = Path(__file__).parent.parent.parent
        
        required_dirs = [
            "src",
            "src/shared", 
            "src/projects/fruitfly_aging",
            "configs",
            "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            
        # Check for key files
        key_files = [
            "run_timeflies.py",
            "pyproject.toml"
        ]
        
        for file_name in key_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"