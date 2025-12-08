import os
import shutil
from pathlib import Path

# Proje root
ROOT = Path("/Users/cemresudeakdag/TİROİD_KANSER_TESPİTİ")

# Taşınacak dosyalar
MOVES = {
    # Models
    "model.py": "src/models/vae.py",
    "classifier.py": "src/models/classifier.py",
    
    # Data
    "data_loader.py": "src/data/dataset.py",
    "organize_data.py": "src/data/organize.py",
    
    # Training
    "train.py": "src/training/train_vae.py",
    "train_classifier.py": "src/training/train_classifier.py",
    "optimize.py": "src/training/optimize.py",
    
    # Evaluation
    "test.py": "src/evaluation/test_vae.py",
    "hybrid_test.py": "src/evaluation/test_hybrid_v2.py",
    "hybrid_test_v3.py": "src/evaluation/test_hybrid_v3.py",
    
    # Utils
    "config.py": "src/utils/config.py",
    "utils.py": "src/utils/helpers.py",
    
    # Docs
    "COMPREHENSIVE_PROJECT_GUIDE.md": "docs/COMPREHENSIVE_GUIDE.md",
    "COMPLETE_PROJECT_GUIDE.txt": "docs/COMPLETE_GUIDE.txt",
    "PROJECT_DOCUMENTATION.md": "docs/PROJECT_DOCUMENTATION.md",
}

def reorganize():
    print("Reorganizing project structure...")
    
    for old_path, new_path in MOVES.items():
        old_file = ROOT / old_path
        new_file = ROOT / new_path
        
        if old_file.exists():
            # Create directory if not exists
            new_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.copy2(old_file, new_file)
            print(f"✓ Moved: {old_path} → {new_path}")
        else:
            print(f"✗ Not found: {old_path}")
    
    print("\nReorganization complete!")
    print("\nNext steps:")
    print("1. Update import statements in all files")
    print("2. Test if everything works")
    print("3. Remove old files: rm model.py classifier.py ...")

if __name__ == "__main__":
    reorganize()
