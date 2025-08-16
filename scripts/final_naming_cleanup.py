#!/usr/bin/env python3
"""Final cleanup to standardize naming convention in data structure."""

import os
import shutil
from pathlib import Path

def fix_full_data_naming():
    """Change 'full_data' to 'full' to match our standard."""
    print("🔧 Fixing 'full_data' to 'full' naming...")
    
    directories_to_rename = []
    
    for root, dirs, files in os.walk("data/processed"):
        for dir_name in dirs:
            if "full_data" in dir_name:
                old_path = os.path.join(root, dir_name)
                new_name = dir_name.replace("full_data", "full")
                new_path = os.path.join(root, new_name)
                directories_to_rename.append((old_path, new_path))
    
    # Rename directories (from deepest to shallowest to avoid conflicts)
    directories_to_rename.sort(key=lambda x: x[0].count('/'), reverse=True)
    
    for old_path, new_path in directories_to_rename:
        if os.path.exists(old_path):
            print(f"  🔄 {os.path.basename(old_path)} → {os.path.basename(new_path)}")
            shutil.move(old_path, new_path)
    
    print(f"✅ Fixed {len(directories_to_rename)} naming patterns")

def verify_structure():
    """Verify the final data structure is clean."""
    print("\n📋 Verifying final structure...")
    
    # Check raw structure
    raw_h5ad = Path("data/raw/h5ad")
    if raw_h5ad.exists():
        h5ad_count = len(list(raw_h5ad.rglob("*.h5ad")))
        print(f"  ✅ {h5ad_count} h5ad files in data/raw/h5ad/")
    
    # Check gene lists
    gene_lists = Path("data/raw/gene_lists")
    if gene_lists.exists():
        csv_count = len(list(gene_lists.glob("*.csv")))
        print(f"  ✅ {csv_count} gene list CSV files")
    
    # Check processed structure
    processed = Path("data/processed")
    if processed.exists():
        correction_dirs = [d.name for d in processed.iterdir() if d.is_dir()]
        print(f"  ✅ Correction types: {', '.join(correction_dirs)}")
        
        # Count experiments
        experiment_count = 0
        for correction_dir in processed.iterdir():
            if correction_dir.is_dir():
                for exp_dir in correction_dir.iterdir():
                    if exp_dir.is_dir():
                        experiment_count += 1
        
        print(f"  ✅ {experiment_count} experiment types in processed data")

def main():
    """Main cleanup function."""
    print("🔧 Final naming cleanup...")
    print("=" * 50)
    
    try:
        fix_full_data_naming()
        verify_structure()
        
        print("\n" + "=" * 50)
        print("✅ Final cleanup complete!")
        print("\nConsistent data structure:")
        print("📂 data/raw/h5ad/ - Original h5ad files by tissue")
        print("📂 data/raw/gene_lists/ - autosomal.csv, sex.csv")
        print("📂 data/processed/batch_corrected/ - Clean 2-level structure")
        print("📂 data/processed/uncorrected/ - Clean 2-level structure")
        print("\nNaming: tissue-model-encoding/method-cells-sexes")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    main()