#!/usr/bin/env python3
"""Reorganize the newly added data folder to match our clean 2-level convention."""

import os
import shutil
from pathlib import Path

def move_gene_lists():
    """Move gene list files to proper location."""
    print("📋 Moving gene lists...")
    
    gene_files = ["autosomal.csv", "sex.csv"]
    target_dir = Path("data/raw/gene_lists")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for gene_file in gene_files:
        source = Path("data") / gene_file
        target = target_dir / gene_file
        
        if source.exists():
            shutil.move(source, target)
            print(f"  ✅ {gene_file} → {target}")

def organize_h5ad_files():
    """Move h5ad files to clean structure."""
    print("\n🧬 Organizing h5ad files...")
    
    h5ad_dir = Path("data/h5ad")
    raw_dir = Path("data/raw/h5ad")
    
    if h5ad_dir.exists():
        # Move entire h5ad structure to raw
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        shutil.move(h5ad_dir, raw_dir)
        print("  ✅ h5ad files → data/raw/h5ad/")

def reorganize_processed_data():
    """Reorganize the old deep structure to new 2-level convention."""
    print("\n📂 Reorganizing processed data to 2-level structure...")
    
    corrections = ["batch_corrected", "uncorrected"]
    
    for correction in corrections:
        source_dir = Path("data") / correction
        target_dir = Path("data/processed") / correction
        
        if not source_dir.exists():
            continue
            
        print(f"\n  Processing {correction}...")
        
        # Navigate the old structure and reorganize
        for tissue_dir in source_dir.iterdir():
            if not tissue_dir.is_dir():
                continue
                
            tissue = tissue_dir.name
            
            for model_dir in tissue_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_type = model_dir.name.lower()
                
                for encoding_dir in model_dir.iterdir():
                    if not encoding_dir.is_dir():
                        continue
                        
                    encoding = encoding_dir.name
                    
                    for gene_dir in encoding_dir.iterdir():
                        if not gene_dir.is_dir():
                            continue
                            
                        gene_method = gene_dir.name
                        
                        for cell_dir in gene_dir.iterdir():
                            if not cell_dir.is_dir():
                                continue
                                
                            cell_type = cell_dir.name
                            
                            for sex_dir in cell_dir.iterdir():
                                if not sex_dir.is_dir():
                                    continue
                                    
                                sex_type = sex_dir.name
                                
                                # Generate new path with 2-level structure
                                base_experiment = f"{tissue}-{model_type}-{encoding}"
                                
                                # Clean up naming
                                if cell_type == "celltype_all":
                                    cell_clean = "all_cells"
                                else:
                                    cell_clean = cell_type.replace("celltype_", "").replace(" ", "_")
                                
                                if sex_type == "sextype_all":
                                    sex_clean = "all_sexes"
                                else:
                                    sex_clean = sex_type.replace("sextype_", "")
                                
                                config_details = f"{gene_method}-{cell_clean}-{sex_clean}"
                                
                                new_path = target_dir / base_experiment / config_details
                                new_path.mkdir(parents=True, exist_ok=True)
                                
                                # Copy files
                                for file_path in sex_dir.iterdir():
                                    if file_path.is_file():
                                        target_file = new_path / file_path.name
                                        shutil.copy2(file_path, target_file)
                                        print(f"    ✅ {file_path.name} → {new_path}")
        
        # Remove old structure after successful reorganization
        if source_dir.exists():
            shutil.rmtree(source_dir)
            print(f"  🗑️ Removed old {correction} structure")

def main():
    """Main reorganization function."""
    print("🔄 Reorganizing new data structure...")
    print("=" * 50)
    
    try:
        move_gene_lists()
        organize_h5ad_files()
        reorganize_processed_data()
        
        print("\n" + "=" * 50)
        print("✅ Data reorganization complete!")
        print("\nNew clean structure:")
        print("📂 data/raw/h5ad/ - Original h5ad files")
        print("📂 data/raw/gene_lists/ - Gene list CSVs")
        print("📂 data/processed/ - Preprocessed data with 2-level naming")
        
    except Exception as e:
        print(f"❌ Error during reorganization: {e}")

if __name__ == "__main__":
    main()