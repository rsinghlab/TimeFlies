#!/usr/bin/env python3
"""
TimeFlies Setup - Initial Data Splitting

This must be run FIRST before training any models.
It creates train/test splits for the datasets.

Usage:
    python run_setup.py
    python run_setup.py --config configs/my_config.yaml
"""

import sys
from pathlib import Path

# Add src to Python path for imports
current_dir = Path(__file__).parent.absolute()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from timeflies.core.config_manager import ConfigManager
    from timeflies.data.preprocessing.setup import DataSetupManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the TimeFlies root directory.")
    sys.exit(1)


def main():
    """Run the data setup process."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TimeFlies: Initial data splitting setup"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file (default: configs/default.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config_manager = ConfigManager(args.config)
        else:
            config_manager = ConfigManager()  # Uses default config
        
        config = config_manager.get_config()
        
        # Initialize and run setup
        print("Starting TimeFlies data setup...")
        print(f"Using tissue: {config.data.tissue}")
        print(f"Batch correction: {config.data.batch_correction.enabled}")
        
        setup_manager = DataSetupManager(config)
        setup_manager.run()
        
        print("Data setup completed successfully!")
        print("You can now run: python run_timeflies.py [options]")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()