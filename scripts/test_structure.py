#!/usr/bin/env python3
"""Simple test script for the refactored TimeFlies structure."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test basic imports that don't require scientific packages."""
    print("Testing basic imports...")
    
    try:
        # Test utility imports
        from timeflies.utils.logging_config import setup_logging, get_logger
        from timeflies.utils.constants import SUPPORTED_MODEL_TYPES, DEFAULT_N_TOP_GENES
        from timeflies.utils.exceptions import TimeFliesError, ConfigurationError
        print("✅ Utility imports successful")
        
        # Test logging
        setup_logging(level="INFO")
        logger = get_logger("test")
        logger.info("Logging system working")
        print("✅ Logging system working")
        
        # Test constants
        print(f"✅ Constants loaded: {len(SUPPORTED_MODEL_TYPES)} model types")
        
        # Test exceptions
        try:
            raise TimeFliesError("Test error")
        except TimeFliesError as e:
            print("✅ Custom exceptions working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from timeflies.core.config_manager import Config
        
        # Test Config object
        test_config = {"test": {"nested": "value"}}
        config = Config(test_config)
        
        assert config.test.nested == "value"
        print("✅ Configuration object working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_cli_parser():
    """Test CLI argument parsing."""
    print("\nTesting CLI parser...")
    
    try:
        from timeflies.utils.cli_parser import create_parser, parse_args
        
        # Test parser creation
        parser = create_parser()
        
        # Test parsing some arguments
        test_args = ["--model-type", "cnn", "--tissue", "head", "--verbose"]
        args = parse_args(test_args)
        
        assert args.model_type == "cnn"
        assert args.tissue == "head"
        assert args.verbose == True
        
        print("✅ CLI parser working")
        return True
        
    except Exception as e:
        print(f"❌ CLI parser test failed: {e}")
        return False

def test_folder_structure():
    """Test that the folder structure is correct."""
    print("\nTesting folder structure...")
    
    src_dir = project_root / "src" / "timeflies"
    
    expected_dirs = [
        "core",
        "data",
        "data/preprocessing", 
        "models",
        "evaluation",
        "analysis",
        "utils"
    ]
    
    expected_files = [
        "__init__.py",
        "core/__init__.py",
        "core/config.py",
        "core/pipeline_manager.py",
        "data/__init__.py",
        "data/loaders.py",
        "data/preprocessing/__init__.py",
        "data/preprocessing/data_processor.py",
        "data/preprocessing/gene_filter.py",
        "utils/__init__.py",
        "utils/logging_config.py",
        "utils/constants.py",
        "utils/exceptions.py",
        "utils/cli_parser.py",
        "models/__init__.py",
        "models/model_factory.py"
    ]
    
    try:
        # Check directories
        for dir_path in expected_dirs:
            full_path = src_dir / dir_path
            if not full_path.exists():
                print(f"❌ Missing directory: {dir_path}")
                return False
        
        # Check files
        for file_path in expected_files:
            full_path = src_dir / file_path
            if not full_path.exists():
                print(f"❌ Missing file: {file_path}")
                return False
        
        print("✅ Folder structure correct")
        return True
        
    except Exception as e:
        print(f"❌ Folder structure test failed: {e}")
        return False

def test_yaml_config():
    """Test YAML configuration loading."""
    print("\nTesting YAML configuration...")
    
    config_path = project_root / "configs" / "default.yaml"
    
    if not config_path.exists():
        print("⚠️ Default YAML config not found (this is OK)")
        return True
    
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Check some expected keys
        expected_keys = ["general", "device", "data", "model"]
        for key in expected_keys:
            if key not in config_data:
                print(f"❌ Missing config section: {key}")
                return False
        
        print("✅ YAML configuration valid")
        return True
        
    except ImportError:
        print("⚠️ PyYAML not installed, skipping YAML test")
        return True
    except Exception as e:
        print(f"❌ YAML config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing TimeFlies Refactored Structure")
    print("=" * 50)
    
    tests = [
        test_folder_structure,
        test_basic_imports,
        test_config_system,
        test_cli_parser,
        test_yaml_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored structure is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)