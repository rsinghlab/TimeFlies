#!/usr/bin/env python3
"""Reorganize h5ad files to put batch corrected and uncorrected in same tissue folder."""

import shutil
from pathlib import Path

def reorganize_h5ad_files():
    """Move h5ad files to flatter structure with both versions together."""
    print("🧬 Reorganizing h5ad files...")
    
    h5ad_root = Path("data/raw/h5ad")
    
    for tissue_dir in h5ad_root.iterdir():
        if not tissue_dir.is_dir() or tissue_dir.name.startswith('.'):
            continue
            
        tissue = tissue_dir.name
        print(f"\n  Processing {tissue}...")
        
        # Collect all h5ad files from subdirectories
        h5ad_files = []
        
        for subdir in tissue_dir.iterdir():
            if subdir.is_dir():
                correction_type = subdir.name
                for h5ad_file in subdir.glob("*.h5ad"):
                    # Create descriptive filename
                    base_name = h5ad_file.stem
                    if correction_type == "batch_corrected":
                        new_name = f"{base_name}_batch.h5ad"
                    else:
                        new_name = f"{base_name}.h5ad"
                    
                    h5ad_files.append((h5ad_file, new_name))
        
        # Move files directly to tissue directory
        for old_file, new_name in h5ad_files:
            target = tissue_dir / new_name
            print(f"    ✅ {old_file.name} → {new_name}")
            shutil.move(old_file, target)
        
        # Remove empty subdirectories
        for subdir in tissue_dir.iterdir():
            if subdir.is_dir():
                try:
                    subdir.rmdir()
                    print(f"    🗑️ Removed empty {subdir.name}/")
                except OSError:
                    print(f"    ⚠️ Could not remove {subdir.name}/ (not empty)")

def verify_h5ad_structure():
    """Verify the new h5ad structure."""
    print("\n📋 Verifying h5ad structure...")
    
    h5ad_root = Path("data/raw/h5ad")
    
    for tissue_dir in h5ad_root.iterdir():
        if not tissue_dir.is_dir() or tissue_dir.name.startswith('.'):
            continue
            
        tissue = tissue_dir.name
        h5ad_files = list(tissue_dir.glob("*.h5ad"))
        
        batch_files = [f for f in h5ad_files if "_batch" in f.name]
        regular_files = [f for f in h5ad_files if "_batch" not in f.name]
        
        print(f"  📂 {tissue}/")
        print(f"    📄 {len(regular_files)} uncorrected h5ad files")
        print(f"    📄 {len(batch_files)} batch corrected h5ad files")

def main():
    """Main reorganization function."""
    print("🔄 Reorganizing h5ad file structure...")
    print("=" * 50)
    
    try:
        reorganize_h5ad_files()
        verify_h5ad_structure()
        
        print("\n" + "=" * 50)
        print("✅ H5AD reorganization complete!")
        print("\nNew structure:")
        print("📂 data/raw/h5ad/head/")
        print("    📄 fly_eval.h5ad (uncorrected)")
        print("    📄 fly_eval_batch.h5ad (batch corrected)")
        print("📂 data/raw/h5ad/body/")
        print("    📄 fly_eval.h5ad (uncorrected)")
        print("    📄 fly_eval_batch.h5ad (batch corrected)")
        
    except Exception as e:
        print(f"❌ Error during reorganization: {e}")

if __name__ == "__main__":
    main()