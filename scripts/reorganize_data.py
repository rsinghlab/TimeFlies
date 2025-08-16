#!/usr/bin/env python3
"""Script to reorganize data into the new cleaner folder structure."""

import os
import shutil
from pathlib import Path

def reorganize_processed_data():
    """Reorganize processed data into new structure."""
    
    # Source and destination
    old_processed = Path("data/preprocessed")
    new_processed = Path("data/processed")
    
    if not old_processed.exists():
        print("No old preprocessed data found to reorganize")
        return
    
    print("📂 Reorganizing processed data...")
    
    # Map old structure to new structure
    for correction_dir in old_processed.iterdir():
        if not correction_dir.is_dir():
            continue
            
        correction_name = correction_dir.name  # batch_corrected or uncorrected
        
        for tissue_dir in correction_dir.iterdir():
            if not tissue_dir.is_dir():
                continue
                
            tissue = tissue_dir.name
            
            for model_dir in tissue_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_type = model_dir.name
                
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
                                
                                # Generate new path
                                base_experiment = f"{tissue}-{model_type.lower()}-{encoding}"
                                config_details = f"{gene_method}-{cell_type}-{sex_type}"
                                
                                new_path = new_processed / correction_name / base_experiment / config_details
                                new_path.mkdir(parents=True, exist_ok=True)
                                
                                # Copy files
                                for file_path in sex_dir.iterdir():
                                    if file_path.is_file():
                                        shutil.copy2(file_path, new_path / file_path.name)
                                        print(f"  ✅ {file_path.name} → {new_path}")

def reorganize_models():
    """Reorganize model outputs into new structure."""
    
    print("\n🤖 Reorganizing model outputs...")
    
    models_dir = Path("outputs/models")
    
    for correction_dir in models_dir.iterdir():
        if not correction_dir.is_dir():
            continue
            
        correction_name = correction_dir.name
        temp_dir = Path(f"temp_{correction_name}")
        
        # Create temporary directory to reorganize
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        for tissue_dir in correction_dir.iterdir():
            if not tissue_dir.is_dir():
                continue
                
            tissue = tissue_dir.name
            
            for model_dir in tissue_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_type = model_dir.name
                
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
                                
                                # Generate new path
                                base_experiment = f"{tissue}-{model_type.lower()}-{encoding}"
                                config_details = f"{gene_method}-{cell_type}-{sex_type}"
                                
                                new_path = temp_dir / base_experiment / config_details
                                new_path.mkdir(parents=True, exist_ok=True)
                                
                                # Move files
                                for file_path in sex_dir.iterdir():
                                    if file_path.is_file():
                                        shutil.copy2(file_path, new_path / file_path.name)
                                        print(f"  ✅ {file_path.name} → {new_path}")
        
        # Replace old structure with new
        shutil.rmtree(correction_dir)
        shutil.move(temp_dir, correction_dir)

def reorganize_results():
    """Reorganize analysis results into new structure."""
    
    print("\n📊 Reorganizing analysis results...")
    
    results_dir = Path("outputs/results")
    
    for correction_dir in results_dir.iterdir():
        if not correction_dir.is_dir():
            continue
            
        correction_name = correction_dir.name
        temp_dir = Path(f"temp_results_{correction_name}")
        
        # Create temporary directory to reorganize
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        for tissue_dir in correction_dir.iterdir():
            if not tissue_dir.is_dir():
                continue
                
            tissue = tissue_dir.name
            
            for model_dir in tissue_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_type = model_dir.name
                
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
                                
                                # Generate new path
                                base_experiment = f"{tissue}-{model_type.lower()}-{encoding}"
                                config_details = f"{gene_method}-{cell_type}-{sex_type}"
                                
                                new_path = temp_dir / base_experiment / config_details
                                new_path.mkdir(parents=True, exist_ok=True)
                                
                                # Copy entire Results directory structure
                                for item in sex_dir.rglob("*"):
                                    if item.is_file():
                                        relative_path = item.relative_to(sex_dir)
                                        target_path = new_path / relative_path
                                        target_path.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(item, target_path)
                                        print(f"  ✅ {relative_path} → {target_path}")
        
        # Replace old structure with new
        shutil.rmtree(correction_dir)
        shutil.move(temp_dir, correction_dir)

def main():
    """Main reorganization function."""
    print("🔄 Reorganizing TimeFlies data structure...")
    print("=" * 50)
    
    try:
        reorganize_processed_data()
        reorganize_models() 
        reorganize_results()
        
        print("\n" + "=" * 50)
        print("✅ Data reorganization complete!")
        print("\nNew structure:")
        print("📂 data/processed/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/")
        print("🤖 outputs/models/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/")
        print("📊 outputs/results/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/")
        
    except Exception as e:
        print(f"❌ Error during reorganization: {e}")

if __name__ == "__main__":
    main()