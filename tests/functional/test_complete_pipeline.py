"""Functional tests for complete end-to-end pipelines.

These tests run the full workflow from data loading through model training 
to evaluation and interpretation. They use real (small) data and verify 
the entire pipeline works together.
"""

import pytest
import os
import sys
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from projects.fruitfly_aging.core.config_manager import ConfigManager
from shared.core.pipeline_manager import PipelineManager
from shared.utils.path_manager import PathManager
from tests.fixtures.sample_data import create_sample_anndata


@pytest.mark.slow
class TestCompletePipeline:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def complete_project_setup(self):
        """Set up a complete project structure with data and config."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir)
        
        # Create project structure
        (project_dir / "src").mkdir()
        (project_dir / "configs").mkdir()
        (project_dir / "outputs").mkdir()
        data_dir = project_dir / "data" / "fruitfly_aging" / "head"
        data_dir.mkdir(parents=True)
        
        # Create sample data files
        adata_train = create_sample_anndata(n_obs=200, n_vars=100, tissue="head")
        adata_eval = create_sample_anndata(n_obs=100, n_vars=100, tissue="head") 
        adata_original = create_sample_anndata(n_obs=300, n_vars=100, tissue="head")
        
        # Save data files
        train_path = data_dir / "drosophila_head_aging_train.h5ad"
        eval_path = data_dir / "drosophila_head_aging_eval.h5ad"
        original_path = data_dir / "drosophila_head_aging_original.h5ad"
        
        adata_train.write_h5ad(train_path)
        adata_eval.write_h5ad(eval_path)
        adata_original.write_h5ad(original_path)
        
        # Create minimal config with correct structure
        config_data = {
            'general': {
                'project_name': 'test_aging',
                'random_state': 42
            },
            'data': {
                'tissue': 'head',
                'model_type': 'MLP',
                'encoding_variable': 'age',
                'cell_type': 'all',
                'sex_type': 'all',
                'gene_selection': 'all',
                'project': 'fruitfly_aging',
                'batch_correction': {'enabled': False}
            },
            'file_locations': {
                'training_file': 'drosophila_head_aging_train.h5ad',
                'evaluation_file': 'drosophila_head_aging_eval.h5ad',
                'original_file': 'drosophila_head_aging_original.h5ad',
                'batch_corrected_files': {
                    'train': 'drosophila_head_aging_train_batch.h5ad',
                    'eval': 'drosophila_head_aging_eval_batch.h5ad'
                }
            },
            'model': {
                'hidden_layers': [64, 32],
                'dropout_rate': 0.1,
                'epochs': 2,  # Very short for testing
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'evaluation': {
                'interpret': False,  # Skip SHAP for speed
                'visualize': False
            }
        }
        
        config_path = project_dir / "configs" / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        yield {
            'project_dir': project_dir,
            'config_path': config_path,
            'data_paths': {
                'train': train_path,
                'eval': eval_path,
                'original': original_path
            }
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_complete_mlp_workflow(self, complete_project_setup):
        """Test complete MLP training and evaluation workflow."""
        setup = complete_project_setup
        
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(setup['project_dir'])
        
        try:
            # Load configuration
            config_manager = ConfigManager(str(setup['config_path']))
            config = config_manager.get_config()
            
            # Initialize pipeline manager
            with patch.dict(os.environ, {'TIMEFLIES_TEST_MODE': '1'}):
                pipeline = PipelineManager(config)
                
                # Run data setup
                pipeline.setup_data()
                
                # Verify data files exist
                path_manager = PathManager(config)
                train_path = path_manager.get_training_file_path()
                eval_path = path_manager.get_evaluation_file_path()
                
                assert train_path.exists(), f"Training file not found: {train_path}"
                assert eval_path.exists(), f"Evaluation file not found: {eval_path}"
                
                # Run model training
                model_result = pipeline.train_model()
                
                # Verify model was created
                assert model_result is not None
                assert hasattr(model_result, 'history') or isinstance(model_result, dict)
                
                # Run evaluation
                evaluation_result = pipeline.evaluate_model()
                
                # Verify evaluation completed
                assert evaluation_result is not None
                
                # Check outputs were created
                outputs_dir = path_manager.get_outputs_directory()
                assert outputs_dir.exists()
                
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.slow
    def test_complete_cnn_workflow_with_batch_correction(self, complete_project_setup):
        """Test complete CNN workflow with batch correction enabled."""
        setup = complete_project_setup
        
        # Modify config for CNN and batch correction
        with open(setup['config_path'], 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['data']['model_type'] = 'CNN'
        config_data['data']['batch_correction']['enabled'] = True
        config_data['model']['epochs'] = 1  # Even shorter for CNN
        
        with open(setup['config_path'], 'w') as f:
            yaml.dump(config_data, f)
        
        original_cwd = os.getcwd()
        os.chdir(setup['project_dir'])
        
        try:
            # Mock scVI to avoid actual batch correction
            with patch('projects.fruitfly_aging.preprocessing.batch_correction.scvi', None):
                with patch.dict(os.environ, {'TIMEFLIES_TEST_MODE': '1'}):
                    config_manager = ConfigManager(str(setup['config_path']))
                    config = config_manager.get_config()
                    
                    pipeline = PipelineManager(config)
                    
                    # This should handle batch correction gracefully when scVI unavailable
                    pipeline.setup_data()
                    
                    # Verify we can still proceed with uncorrected data
                    path_manager = PathManager(config)
                    train_path = path_manager.get_training_file_path()
                    assert train_path.exists()
                    
                    # Training should work even without batch correction
                    model_result = pipeline.train_model()
                    assert model_result is not None
                    
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling_missing_data(self, complete_project_setup):
        """Test pipeline error handling when data files are missing."""
        setup = complete_project_setup
        
        # Remove training data file
        os.remove(setup['data_paths']['train'])
        
        original_cwd = os.getcwd()
        os.chdir(setup['project_dir'])
        
        try:
            with patch.dict(os.environ, {'TIMEFLIES_TEST_MODE': '1'}):
                config_manager = ConfigManager(str(setup['config_path']))
                config = config_manager.get_config()
                
                pipeline = PipelineManager(config)
                
                # Should handle missing data gracefully
                with pytest.raises((FileNotFoundError, Exception)):
                    pipeline.train_model()
                    
        finally:
            os.chdir(original_cwd)
    
    def test_pipeline_state_management(self, complete_project_setup):
        """Test that pipeline maintains proper state across operations."""
        setup = complete_project_setup
        
        original_cwd = os.getcwd()
        os.chdir(setup['project_dir'])
        
        try:
            with patch.dict(os.environ, {'TIMEFLIES_TEST_MODE': '1'}):
                config_manager = ConfigManager(str(setup['config_path']))
                config = config_manager.get_config()
                
                pipeline = PipelineManager(config)
                
                # Check initial state
                assert pipeline.config == config
                
                # After data setup
                pipeline.setup_data()
                
                # After model training
                model_result = pipeline.train_model()
                
                # Pipeline should maintain consistent state
                assert pipeline.config == config
                assert model_result is not None
                
        finally:
            os.chdir(original_cwd)