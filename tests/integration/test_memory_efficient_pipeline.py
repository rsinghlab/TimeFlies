"""Memory-efficient integration tests using small test fixtures only.

This test module ensures we use ONLY the small test fixtures (139K files)
and never load the large real data files (20GB+).
"""

import pytest
import os
import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.timeflies.core.config_manager import Config
from src.timeflies.data.loaders import DataLoader


@pytest.fixture
def isolated_test_env():
    """Create completely isolated test environment with ONLY small test fixtures."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Store original directory
    original_cwd = os.getcwd()
    
    try:
        # Change to temp directory 
        os.chdir(temp_dir)
        
        # Create minimal data structure
        data_dir = temp_dir / "data" / "raw" / "h5ad" / "head"
        data_dir.mkdir(parents=True)
        
        # Copy ONLY the small test fixtures
        fixtures_dir = Path(original_cwd) / "tests" / "fixtures"
        
        # Copy small test files
        small_files = [
            ("test_data.h5ad", "fly_train.h5ad"),
            ("test_data.h5ad", "fly_eval.h5ad"), 
            ("test_data.h5ad", "fly_original.h5ad"),
            ("test_data_corrected.h5ad", "fly_train_batch.h5ad"),
            ("test_data_corrected.h5ad", "fly_eval_batch.h5ad"),
        ]
        
        for src_name, dest_name in small_files:
            src = fixtures_dir / src_name
            dest = data_dir / dest_name
            if src.exists():
                shutil.copy2(src, dest)
                # Verify file is small
                size_mb = dest.stat().st_size / (1024 * 1024)
                assert size_mb < 1, f"Test file {dest_name} is too large: {size_mb:.1f}MB"
        
        # Create gene lists directory
        gene_dir = temp_dir / "data" / "raw" / "gene_lists"
        gene_dir.mkdir(parents=True)
        
        # Create minimal gene lists
        (gene_dir / "autosomal.csv").write_text("gene1\ngene2\ngene3\n")
        (gene_dir / "sex.csv").write_text("sexgene1\nsexgene2\n")
        
        yield temp_dir
        
    finally:
        # Restore original directory
        os.chdir(original_cwd) 
        # Cleanup
        shutil.rmtree(temp_dir)


@pytest.fixture 
def memory_safe_config():
    """Create config that forces small data usage."""
    config_dict = {
        'general': {
            'project_name': 'TimeFlies',
            'version': '0.2.0',
            'random_state': 42
        },
        'data': {
            'tissue': 'head',
            'model_type': 'CNN',
            'encoding_variable': 'age', 
            'cell_type': 'all',
            'sex_type': 'all',
            'batch_correction': {
                'enabled': False  # CRITICAL: Disable to avoid 20GB files
            },
            'filtering': {
                'include_mixed_sex': False
            },
            'sampling': {
                'num_samples': 50,  # Limit sample size
                'num_variables': 100  # Limit gene count
            },
            'train_test_split': {
                'method': 'random',
                'test_split': 0.2,
                'random_state': 42
            }
        },
        'model': {
            'cnn': {
                'filters': [8],  # Minimal CNN
                'kernel_sizes': [3],
                'strides': [1], 
                'paddings': ['same'],
                'pool_sizes': [2],
                'pool_strides': [2],
                'dense_units': [16],  # Small dense layer
                'dropout_rate': 0.3,
                'activation': 'relu'
            },
            'training': {
                'learning_rate': 0.01,
                'epochs': 1,  # Just 1 epoch for testing
                'batch_size': 8,  # Small batch
                'validation_split': 0.2,
                'early_stopping_patience': 1
            }
        },
        'device': {
            'processor': 'CPU',  # Force CPU to avoid GPU memory issues
            'gpu_memory_growth': False
        },
        'data_processing': {
            'preprocessing': {
                'required': True,
                'save_data': False
            },
            'model_management': {
                'load_model': False
            },
            'exploratory_data_analysis': {
                'enabled': False
            },
            'normalization': {
                'enabled': False
            }
        },
        'gene_preprocessing': {
            'gene_filtering': {
                'remove_sex_genes': False,
                'remove_autosomal_genes': False,
                'only_keep_lnc_genes': False,
                'remove_lnc_genes': False,
                'remove_unaccounted_genes': False,
                'select_batch_genes': False,
                'highly_variable_genes': False
            },
            'gene_balancing': {
                'balance_genes': False,
                'balance_lnc_genes': False
            },
            'gene_shuffle': {
                'shuffle_genes': False
            }
        },
        'feature_importance': {
            'run_interpreter': False,
            'run_visualization': False
        },
        'file_locations': {
            'training_file': 'fly_train.h5ad',
            'evaluation_file': 'fly_eval.h5ad', 
            'original_file': 'fly_original.h5ad',
            'batch_corrected_files': {
                'train': 'fly_train_batch.h5ad',
                'eval': 'fly_eval_batch.h5ad'
            }
        }
    }
    
    return Config(config_dict)


class TestMemoryEfficientPipeline:
    """Test pipeline with strict memory limits using only small test fixtures."""
    
    @pytest.mark.integration
    def test_data_loader_uses_small_fixtures_only(self, isolated_test_env, memory_safe_config):
        """Verify DataLoader uses only small test fixtures, not real data."""
        print(f"\n🧪 Testing in isolated directory: {isolated_test_env}")
        
        # Verify we're in the right place and files are small
        data_dir = isolated_test_env / "data" / "raw" / "h5ad" / "head"
        for filename in ["fly_train.h5ad", "fly_eval.h5ad"]:
            filepath = data_dir / filename
            assert filepath.exists(), f"Test fixture {filename} not found"
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"📁 {filename}: {size_mb:.2f}MB")
            assert size_mb < 1, f"File {filename} too large: {size_mb:.2f}MB"
        
        # Create DataLoader and verify it loads small data
        loader = DataLoader(memory_safe_config)
        adata, adata_eval, adata_original = loader.load_data()
        
        # Verify loaded data is small (from our test fixtures)
        assert adata.n_obs == 100, f"Expected 100 cells, got {adata.n_obs}"
        assert adata.n_vars == 500, f"Expected 500 genes, got {adata.n_vars}"
        
        print(f"✅ Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
        print("✅ Data loading test passed - using small fixtures only")
    
    @pytest.mark.integration
    def test_batch_correction_disabled_avoids_large_files(self, isolated_test_env, memory_safe_config):
        """Test that batch correction disabled prevents loading 20GB+ files."""
        print(f"\n🧪 Testing batch correction disabled")
        
        # Ensure batch correction is disabled
        assert memory_safe_config.data.batch_correction.enabled == False
        
        loader = DataLoader(memory_safe_config)
        
        # Load corrected data - should still use small files  
        adata_corrected, adata_eval_corrected = loader.load_corrected_data()
        
        # Verify corrected data is also from small fixtures
        assert adata_corrected.n_obs == 100
        assert adata_corrected.n_vars == 500
        
        print(f"✅ Corrected data: {adata_corrected.n_obs} cells, {adata_corrected.n_vars} genes")
        print("✅ Batch correction test passed - using small files only")
    
    @pytest.mark.integration  
    def test_memory_usage_stays_reasonable(self, isolated_test_env, memory_safe_config):
        """Test that memory usage stays under reasonable limits."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"\n🧠 Initial memory: {initial_memory_mb:.1f}MB")
        
        # Load data
        loader = DataLoader(memory_safe_config)
        adata, adata_eval, adata_original = loader.load_data()
        adata_corrected, adata_eval_corrected = loader.load_corrected_data()
        
        # Check memory after loading
        final_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = final_memory_mb - initial_memory_mb
        
        print(f"🧠 Final memory: {final_memory_mb:.1f}MB")
        print(f"🧠 Memory increase: {memory_increase_mb:.1f}MB")
        
        # Memory increase should be reasonable (< 500MB for small test data)
        assert memory_increase_mb < 500, f"Memory increase too large: {memory_increase_mb:.1f}MB"
        
        print("✅ Memory usage test passed - stayed under 500MB increase")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])