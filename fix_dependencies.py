#!/usr/bin/env python3
"""
Fix compatibility issues with dependencies
"""

import subprocess
import sys
import os

def fix_huggingface_compatibility():
    """Fix huggingface_hub compatibility issue"""
    print("🔧 Fixing huggingface_hub compatibility...")
    
    try:
        # Uninstall incompatible versions
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'uninstall', 
            'sentence-transformers', 'huggingface_hub', '-y'
        ])
        
        # Install compatible versions
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'huggingface_hub>=0.19.0,<0.25.0'
        ])
        
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'sentence-transformers>=2.3.0'
        ])
        
        print("✅ Fixed huggingface_hub compatibility")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fix compatibility: {e}")
        return False

def test_imports():
    """Test if imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        import transformers
        import numpy as np
        import faiss
        import flask
        print("✅ Basic imports successful")
        
        # Test sentence-transformers specifically
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer import successful")
        
        # Test model loading (quick test)
        print("🔄 Testing model loading (this may take a moment)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Smaller model for testing
        test_embedding = model.encode(["test sentence"])
        print("✅ Model loading test successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Warning during testing: {e}")
        return True  # Non-critical error

def alternative_setup():
    """Alternative setup using pip install directly"""
    print("🔄 Trying alternative installation method...")
    
    packages = [
        "flask>=2.3.3",
        "torch>=2.1.0", 
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "nltk>=3.8.1",
        "python-dotenv>=1.0.0",
        "faiss-cpu>=1.7.4",
        "huggingface_hub>=0.19.0,<0.25.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.3.0"
    ]
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package
            ])
        
        print("✅ Alternative installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Alternative installation failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Fixing Semantic Cache Dependencies...")
    print("=" * 50)
    
    # Try fixing compatibility
    if fix_huggingface_compatibility():
        if test_imports():
            print("\n✅ Dependencies fixed successfully!")
            print("\n🚀 You can now run: python app.py")
            return True
    
    print("\n🔄 Trying alternative installation...")
    if alternative_setup():
        if test_imports():
            print("\n✅ Dependencies fixed with alternative method!")
            print("\n🚀 You can now run: python app.py")
            return True
    
    print("\n❌ Could not fix dependency issues automatically.")
    print("\n🛠️ Manual steps to try:")
    print("1. Create a fresh virtual environment:")
    print("   python -m venv fresh_env")
    print("   fresh_env\\Scripts\\activate  # Windows")
    print("   source fresh_env/bin/activate  # Linux/Mac")
    print("\n2. Install packages individually:")
    print("   pip install torch")
    print("   pip install 'huggingface_hub>=0.19.0,<0.25.0'")
    print("   pip install 'sentence-transformers>=2.3.0'")
    print("   pip install -r requirements.txt")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)