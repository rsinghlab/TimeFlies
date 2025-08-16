#!/usr/bin/env python3
"""
TimeFlies - Main Entry Point

Simple, unified entry point for the TimeFlies pipeline.
Choose between original code or new refactored structure.
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point with choice between old and new structure."""
    
    print("🧬 TimeFlies - Time-series Analysis for Biological Data")
    print("=" * 55)
    
    # Check what's available
    original_available = Path("Code/time_flies.py").exists()
    new_available = Path("src/timeflies").exists()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        print("\nAvailable options:")
        if original_available:
            print("  1. original  - Use original Code/ structure")
        if new_available:
            print("  2. new       - Use refactored src/timeflies/ structure") 
            print("  3. test      - Test the new structure")
        
        print("\nUsage:")
        print("  python run_timeflies.py original    # Use original code")
        print("  python run_timeflies.py new         # Use new structure")
        print("  python run_timeflies.py test        # Test new structure")
        print("  python run_timeflies.py new --help  # See all CLI options")
        return

    if choice == "original":
        if not original_available:
            print("❌ Original code not found in Code/ directory")
            return
        
        print("🔄 Running original TimeFlies pipeline...")
        # Add Code directory to path and run original
        sys.path.insert(0, "Code")
        try:
            import time_flies
            # Run original main function if it exists
            if hasattr(time_flies, 'main'):
                time_flies.main()
            else:
                print("Original pipeline ready. Import and run as needed.")
        except Exception as e:
            print(f"❌ Error running original pipeline: {e}")

    elif choice == "new":
        if not new_available:
            print("❌ New structure not found in src/ directory")
            return
        
        print("🚀 Running new TimeFlies pipeline...")
        # Run the enhanced pipeline with remaining arguments
        remaining_args = sys.argv[2:] if len(sys.argv) > 2 else []
        
        # Add src to path
        sys.path.insert(0, "src")
        
        try:
            # Import and run the enhanced script
            os.chdir(Path(__file__).parent)
            exec_path = "scripts/run_pipeline_enhanced.py"
            
            if remaining_args:
                os.system(f"python {exec_path} {' '.join(remaining_args)}")
            else:
                os.system(f"python {exec_path}")
                
        except Exception as e:
            print(f"❌ Error running new pipeline: {e}")

    elif choice == "test":
        print("🧪 Testing new structure...")
        os.system("python scripts/test_structure.py")

    else:
        print(f"❌ Unknown option: {choice}")
        print("Use 'original', 'new', or 'test'")

if __name__ == "__main__":
    main()