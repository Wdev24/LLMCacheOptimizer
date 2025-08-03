#!/usr/bin/env python3
"""
Simplified setup script that skips problematic verification
"""

import os
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
            },
            {
                "query": "How does neural network work?",
                "response": "A neural network works by mimicking the structure and function of biological neurons, using interconnected nodes organized in layers to process information and learn patterns."
            },
            {
                "query": "What is LLM?",
                "response": "LLM stands for Large Language Model, which is a type of AI model trained on vast amounts of text data to understand and generate human-like language."
            }
        ]
        
        with open(cache_file, 'w') as f:
            json.dump(example_data, f, indent=2)
        
        print("✅ Created example cache file")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ Could not download NLTK data: {e}")

def main():
    """Main setup function"""
    print("🚀 Simple Setup for Semantic Cache System...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Create example cache
    create_example_cache()
    
    print("\n" + "=" * 50)
    print("✅ Basic setup completed!")
    print("\n🔧 To fix dependency issues, run:")
    print("   python fix_dependencies.py")
    print("\n🚀 Or try running directly:")
    print("   python app.py")
    print("\n📝 Note: Models will be downloaded on first use")

if __name__ == "__main__":
    main()