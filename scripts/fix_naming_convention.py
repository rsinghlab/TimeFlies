#!/usr/bin/env python3
"""Fix naming convention by replacing spaces with underscores in cell type names."""

import os
import shutil
from pathlib import Path

def fix_cell_type_naming():
    """Fix cell type names by replacing spaces with underscores."""
    
    print("🔧 Fixing cell type naming convention...")
    
    # Find all directories that need renaming
    directories_to_rename = []
    
    for root, dirs, files in os.walk("outputs"):
        for dir_name in dirs:
            if any(cell_type in dir_name for cell_type in [
                "epithelial cell", "muscle cell", "glial cell", 
                "cns neuron", "sensory neuron"
            ]):
                old_path = os.path.join(root, dir_name)
                new_name = dir_name.replace("epithelial cell", "epithelial_cell")
                new_name = new_name.replace("muscle cell", "muscle_cell")
                new_name = new_name.replace("glial cell", "glial_cell")
                new_name = new_name.replace("cns neuron", "cns_neuron")
                new_name = new_name.replace("sensory neuron", "sensory_neuron")
                new_path = os.path.join(root, new_name)
                
                if old_path != new_path:
                    directories_to_rename.append((old_path, new_path))
    
    # Rename directories (start from deepest to avoid path conflicts)
    directories_to_rename.sort(key=lambda x: x[0].count('/'), reverse=True)
    
    for old_path, new_path in directories_to_rename:
        if os.path.exists(old_path):
            print(f"  🔄 {old_path} → {new_path}")
            shutil.move(old_path, new_path)
    
    print(f"✅ Fixed {len(directories_to_rename)} directory names")

def fix_old_naming_patterns():
    """Fix any remaining old naming patterns."""
    
    print("\n🔧 Fixing old naming patterns...")
    
    directories_to_rename = []
    
    for root, dirs, files in os.walk("outputs"):
        for dir_name in dirs:
            old_path = os.path.join(root, dir_name)
            new_name = dir_name
            
            # Fix old patterns
            if "celltype_" in dir_name:
                new_name = new_name.replace("celltype_all", "all_cells")
                new_name = new_name.replace("celltype_", "")
            
            if "sextype_" in dir_name:
                new_name = new_name.replace("sextype_all", "all_sexes")
                new_name = new_name.replace("sextype_", "")
            
            # Fix data patterns
            if "full_data" in dir_name:
                new_name = new_name.replace("full_data", "full")
            
            new_path = os.path.join(root, new_name)
            
            if old_path != new_path:
                directories_to_rename.append((old_path, new_path))
    
    # Rename directories
    directories_to_rename.sort(key=lambda x: x[0].count('/'), reverse=True)
    
    for old_path, new_path in directories_to_rename:
        if os.path.exists(old_path):
            print(f"  🔄 {old_path} → {new_path}")
            shutil.move(old_path, new_path)
    
    print(f"✅ Fixed {len(directories_to_rename)} old naming patterns")

def main():
    """Main function to fix all naming conventions."""
    print("🔧 Fixing naming conventions...")
    print("=" * 50)
    
    try:
        fix_cell_type_naming()
        fix_old_naming_patterns()
        
        print("\n" + "=" * 50)
        print("✅ Naming convention fixes complete!")
        print("\nConsistent naming:")
        print("📁 all_cells, epithelial_cell, muscle_cell, etc.")
        print("📁 all_sexes, male, female")
        print("📁 hvg, full, no_sex, balanced, etc.")
        
    except Exception as e:
        print(f"❌ Error fixing naming: {e}")

if __name__ == "__main__":
    main()