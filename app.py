"""
Main Flask Application Entry Point
Hybrid Semantic Caching System
"""

from flask import Flask
import logging
import os
from routes.main import main_bp
from core.cache_engine import SemanticCacheEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'semantic-cache-dev-key')
    
    # Register blueprints
    app.register_blueprint(main_bp)
    
    return app

def initialize_cache_system():
    """Initialize the semantic cache system on startup"""
    try:
        cache_engine = SemanticCacheEngine()
        cache_engine.initialize()
        logging.info("✅ Semantic Cache System initialized successfully")
        return cache_engine
    except Exception as e:
        logging.error(f"❌ Failed to initialize cache system: {e}")
        raise

def initialize_enhanced_cache_system(cache_engine):
    """Initialize the enhanced semantic cache system"""
    try:
        from enhanced_multi_stage_cache import EnhancedSemanticCacheEngine
        
        # Create enhanced engine using existing components
        enhanced_engine = EnhancedSemanticCacheEngine(
            embedding_model=cache_engine.embedding_model,
            reranker_model=cache_engine.reranker,
            faiss_manager=cache_engine.faiss_manager,
            llm_client=cache_engine.llm_client,
            entity_extractor=getattr(cache_engine, 'entity_extractor', None)
        )
        
        logging.info("✅ Enhanced Semantic Cache System initialized successfully")
        return enhanced_engine
        
    except ImportError as e:
        logging.warning(f"⚠️ Enhanced cache not available: {e}")
        logging.warning("⚠️ Make sure enhanced_multi_stage_cache.py is in the project directory")
        return None
    except Exception as e:
        logging.error(f"❌ Failed to initialize enhanced cache: {e}")
        return None

if __name__ == '__main__':
    # Initialize the regular cache system
    cache_engine = initialize_cache_system()
    
    # Initialize the enhanced cache system
    enhanced_cache_engine = initialize_enhanced_cache_system(cache_engine)
    
    # Create Flask app
    app = create_app()
    
    # Store both engines in app context
    app.cache_engine = cache_engine
    
    if enhanced_cache_engine:
        app.enhanced_cache_engine = enhanced_cache_engine
        print("🚀 Enhanced cache engine loaded!")
        print("🔬 Advanced features available at /api/query/enhanced")
    else:
        print("⚠️ Running with regular cache engine only")
        print("📝 To enable enhanced features, ensure enhanced_multi_stage_cache.py is present")
    
    print("🚀 Starting Hybrid Semantic Cache System...")
    print("📍 Access the system at: http://localhost:5000")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )