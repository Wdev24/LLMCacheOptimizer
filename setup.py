#!/usr/bin/env python3
"""
Setup script for Hybrid Semantic Cache System
Creates necessary directories and initializes the system
"""

import os
import sys
import subprocess
import json

def create_directories():
    """Create necessary directories"""
    directories = [
        'routes',
        'models', 
        'core',
        'llm',
        'templates',
        'data',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory in ['routes', 'models', 'core', 'llm']:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Package initialization file\n')
    
    # Create .gitkeep files for empty directories
    for directory in ['data', 'logs']:
        gitkeep_file = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, 'w') as f:
                f.write('')
    
    print("✅ Created directory structure")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ Could not download NLTK data: {e}")

def create_example_cache():
    """Create example cache file if it doesn't exist"""
    cache_file = 'data/example_cache.json'
    if not os.path.exists(cache_file):
        example_data = [
            {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve performance through experience without being explicitly programmed."
            },
            {
                "query": "Define artificial intelligence", 
                "response": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that can think, learn, and problem-solve like humans."
            }
        ]
        
        with open(cache_file, 'w') as f:
            json.dump(example_data, f, indent=2)
        
        print("✅ Created example cache file")

def verify_installation():
    """Verify that all components can be imported"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import flask
        import numpy as np
        import sklearn
        import requests
        
        print("✅ All core dependencies verified")
        
        # Test model loading (this might take a while on first run)
        print("🧠 Testing model loading (this may take a few minutes on first run)...")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        test_embedding = model.encode(["test sentence"])
        
        print("✅ Embedding model test successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Warning: {e}")
        return True  # Non-critical error

def main():
    """Main setup function"""
    print("🚀 Setting up Hybrid Semantic Cache System...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed due to dependency installation issues")
        return False
    
    # Download NLTK data
    download_nltk_data()
    
    # Create example cache
    create_example_cache()
    
    # Verify installation
    if verify_installation():
        print("\n" + "=" * 50)
        print("✅ Setup completed successfully!")
        print("\n🚀 To start the system, run:")
        print("   python app.py")
        print("\n🌐 Then open your browser to:")
        print("   http://localhost:5000")
        print("\n📚 The system will download models on first use (may take a few minutes)")
        return True
    else:
        print("\n❌ Setup completed with warnings. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)