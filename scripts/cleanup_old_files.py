#!/usr/bin/env python3
"""Script to clean up old/outdated files after refactoring."""

import shutil
from pathlib import Path

def cleanup_old_directories():
    """Remove old directory structures that are now outdated."""
    
    print("🧹 Cleaning up old directories...")
    
    old_dirs_to_remove = [
        "Data",  # Moved to data/raw/
        "Models",  # Moved to outputs/models/
        "Analysis",  # Moved to outputs/results/
        "Code",  # Replaced by src/timeflies/
        "data/preprocessed",  # Reorganized to data/processed/
        "data/uncorrected",  # Old structure
        "Other",  # Miscellaneous old files
    ]
    
    for dir_path in old_dirs_to_remove:
        path = Path(dir_path)
        if path.exists():
            print(f"  🗑️ Removing {dir_path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            print(f"  ✅ {dir_path} already removed")

def cleanup_old_scripts():
    """Remove old script files that are now replaced."""
    
    print("\n🧹 Cleaning up old scripts...")
    
    old_scripts_to_remove = [
        "scripts/run_pipeline.py",  # Replaced by run_timeflies.py
        "scripts/run_pipeline_enhanced.py",  # Integrated into run_timeflies.py
    ]
    
    for script_path in old_scripts_to_remove:
        path = Path(script_path)
        if path.exists():
            print(f"  🗑️ Removing {script_path}")
            path.unlink()
        else:
            print(f"  ✅ {script_path} already removed")

def cleanup_documentation():
    """Remove old documentation files that are now outdated."""
    
    print("\n🧹 Cleaning up old documentation...")
    
    old_docs_to_remove = [
        "REFACTOR_README.md",  # Replaced by NEW_FOLDER_STRUCTURE.md
        "REFACTORING_COMPLETE.md",  # Temporary file
    ]
    
    for doc_path in old_docs_to_remove:
        path = Path(doc_path)
        if path.exists():
            print(f"  🗑️ Removing {doc_path}")
            path.unlink()
        else:
            print(f"  ✅ {doc_path} already removed")

def cleanup_temp_files():
    """Remove temporary files created during reorganization."""
    
    print("\n🧹 Cleaning up temporary files...")
    
    # Find and remove any temp files
    for temp_file in Path(".").glob("temp_*"):
        if temp_file.exists():
            print(f"  🗑️ Removing {temp_file}")
            if temp_file.is_dir():
                shutil.rmtree(temp_file)
            else:
                temp_file.unlink()

def create_gitignore_updates():
    """Update .gitignore for new structure."""
    
    print("\n📝 Updating .gitignore...")
    
    gitignore_additions = [
        "# TimeFlies specific ignores",
        "outputs/logs/*.log",
        "data/processed/*/",
        "*.pkl",
        "*.h5",
        "__pycache__/",
        ".pytest_cache/",
        "temp_*/",
    ]
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""
    
    # Add new ignores if not already present
    additions_to_add = []
    for addition in gitignore_additions:
        if addition not in existing_content:
            additions_to_add.append(addition)
    
    if additions_to_add:
        with open(gitignore_path, 'a') as f:
            f.write("\n\n" + "\n".join(additions_to_add))
        print(f"  ✅ Added {len(additions_to_add)} new ignores to .gitignore")
    else:
        print("  ✅ .gitignore already up to date")

def main():
    """Main cleanup function."""
    print("🧹 Cleaning up outdated files and directories...")
    print("=" * 50)
    
    try:
        cleanup_old_directories()
        cleanup_old_scripts() 
        cleanup_documentation()
        cleanup_temp_files()
        create_gitignore_updates()
        
        print("\n" + "=" * 50)
        print("✅ Cleanup complete!")
        print("\nRemaining structure:")
        print("📂 data/raw/ - Input data")
        print("📂 data/processed/ - Preprocessed data")  
        print("🤖 outputs/models/ - Trained models")
        print("📊 outputs/results/ - Analysis results")
        print("🔧 src/timeflies/ - Refactored code")
        print("🎯 run_timeflies.py - Main entry point")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    main()