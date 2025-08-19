"""
TimeFlies CLI Commands

This module contains all command-line interface commands for TimeFlies.
Each command is implemented as a separate function with proper error handling.
"""

import sys
from pathlib import Path

from ..core.config_manager import ConfigManager
from ..core.pipeline_manager import PipelineManager

# Optional import for batch correction
try:
    from ..data.preprocessing.batch_correction import BatchCorrector
    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    BatchCorrector = None


def validate_environment() -> bool:
    """Validate that the TimeFlies environment is properly set up."""
    try:
        # Check if we can import core modules
        from ..core.config_manager import ConfigManager
        from ..core.pipeline_manager import PipelineManager
        return True
    except ImportError as e:
        print(f"Environment validation failed: {e}")
        return False


def test_command(args) -> int:
    """Run basic system tests to verify installation."""
    print("Running TimeFlies system tests...")
    
    if not validate_environment():
        print("FAILED: Environment validation")
        return 1
    
    print("PASSED: Environment validation")
    
    # Test configuration loading
    try:
        config_path = Path("configs/head_cnn_config.yaml")
        if config_path.exists():
            config_manager = ConfigManager(str(config_path))
            config = config_manager.get_config()
            print("PASSED: Configuration loading")
        else:
            print("WARNING: No sample config found, creating minimal config test")
            config_manager = ConfigManager()
            config = config_manager.get_config()
            print("PASSED: Configuration creation")
    except Exception as e:
        print(f"FAILED: Configuration test - {e}")
        return 1
    
    print("All basic tests passed!")
    print("\nNext steps:")
    print("1. Ensure your data is in the correct directory structure")
    print("2. Review configuration files in configs/")
    print("3. Run: python run_timeflies.py train --help")
    
    return 0


def setup_command(args) -> int:
    """Set up data and prepare for analysis."""
    print("Setting up TimeFlies data structure...")
    
    try:
        # Create necessary directories
        directories = [
            "data/raw/h5ad",
            "data/processed", 
            "outputs/models",
            "outputs/results",
            "outputs/logs"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your .h5ad files in data/raw/h5ad/")
        print("2. Review and customize configs/head_cnn_config.yaml")
        print("3. Run training with: python run_timeflies.py train")
        
        return 0
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return 1


def train_command(args, config) -> int:
    """Train a model with the specified configuration."""
    try:
        # Override config with command line arguments
        if hasattr(args, 'tissue') and args.tissue:
            config.data.tissue = args.tissue
        if hasattr(args, 'model') and args.model:
            config.data.model_type = args.model.upper()
        if hasattr(args, 'target') and args.target:
            config.data.encoding_variable = args.target
        # Handle batch correction flags
        if hasattr(args, 'batch_correction') and args.batch_correction:
            config.data.batch_correction.enabled = True
        elif hasattr(args, 'no_batch_correction') and args.no_batch_correction:
            config.data.batch_correction.enabled = False
        # Otherwise keep the config file default
        if hasattr(args, 'cell_type') and args.cell_type:
            config.data.cell_type = args.cell_type
        if hasattr(args, 'sex_type') and args.sex_type:
            config.data.sex_type = args.sex_type
        
        print(f"Starting training with:")
        print(f"  Tissue: {config.data.tissue}")
        print(f"  Model: {config.data.model_type}")
        print(f"  Target: {config.data.encoding_variable}")
        print(f"  Batch correction: {config.data.batch_correction.enabled}")
        
        # Initialize and run pipeline
        pipeline = PipelineManager(config)
        results = pipeline.run()
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {results.get('model_path', 'outputs/models/')}")
        print(f"Results saved to: {results.get('results_path', 'outputs/results/')}")
        if 'duration' in results:
            print(f"Training duration: {results['duration']:.1f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def batch_command(args) -> int:
    """Run batch correction pipeline."""
    if not BATCH_CORRECTION_AVAILABLE:
        print("Error: Cannot perform batch correction.")
        print("The scVI dependencies are not installed in this environment.")
        print("To perform batch correction, install: pip install scvi-tools scanpy scib")
        print("Note: You can still use existing batch-corrected data files.")
        return 1
    
    print("Running batch correction...")
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config) if args.config else ConfigManager()
        config = config_manager.get_config()
        
        # Initialize batch corrector
        batch_corrector = BatchCorrector(config)
        
        if args.train:
            print("Training batch correction model...")
            batch_corrector.train()
        else:
            print("Applying batch correction...")
            batch_corrector.correct()
        
        print("Batch correction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Batch correction failed: {e}")
        return 1


def evaluate_command(args, config=None) -> int:
    """Evaluate a trained model."""
    print("Evaluating model...")
    
    try:
        # Use provided config or load from args
        if config is None:
            config_manager = ConfigManager(args.config) if hasattr(args, 'config') and args.config else ConfigManager()
            config = config_manager.get_config()
        
        # Override config with command line arguments
        if hasattr(args, 'tissue') and args.tissue:
            config.data.tissue = args.tissue
        
        # Override model loading - force to True for evaluation
        if not hasattr(config, 'data_processing'):
            from types import SimpleNamespace
            config.data_processing = SimpleNamespace()
        if not hasattr(config.data_processing, 'model_management'):
            config.data_processing.model_management = SimpleNamespace()
        config.data_processing.model_management.load_model = True
        
        print(f"Evaluating model with:")
        print(f"  Tissue: {config.data.tissue}")
        print(f"  Model: {config.data.model_type}")
        print(f"  Target: {config.data.encoding_variable}")
        
        # Initialize pipeline in evaluation mode
        pipeline = PipelineManager(config)
        pipeline.setup_gpu()
        pipeline.load_data()
        pipeline.run_preprocessing()
        pipeline.load_or_train_model()
        pipeline.run_metrics()
        
        if hasattr(args, 'interpret') and args.interpret:
            pipeline.run_interpretation()
        
        if hasattr(args, 'visualize') and args.visualize:
            pipeline.run_visualizations()
        
        print("Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def validate_command(args) -> int:
    """Validate configuration files."""
    try:
        config_path = args.config or "configs/head_cnn_config.yaml"
        
        print(f"Validating configuration: {config_path}")
        
        # Try to load the configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        # Validate required fields
        required_fields = [
            'data.tissue',
            'data.model_type', 
            'data.encoding_variable'
        ]
        
        for field in required_fields:
            value = config
            for part in field.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    print(f"MISSING: Required field {field}")
                    return 1
        
        print("Configuration validation passed!")
        print(f"  Tissue: {config.data.tissue}")
        print(f"  Model: {config.data.model_type}")
        print(f"  Target: {config.data.encoding_variable}")
        
        return 0
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1


