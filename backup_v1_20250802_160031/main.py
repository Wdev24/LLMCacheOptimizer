"""
Main API Routes for Semantic Cache System
"""

from flask import Blueprint, render_template, request, jsonify, current_app
import logging
import time

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@main_bp.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@main_bp.route('/api/query', methods=['POST'])
def process_query():
    """
    Process user query through semantic cache system
    Returns: JSON response with result, source, confidence, and timing
    """
    try:
        start_time = time.time()
        
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'No query provided',
                'status': 'error'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'error': 'Empty query provided',
                'status': 'error'
            }), 400
        
        logger.info(f"🔍 Processing query: '{query}'")
        
        # Get cache engine from app context
        cache_engine = current_app.cache_engine
        
        # Process query through semantic cache
        result = cache_engine.process_query(query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'query': query,
            'response': result['response'],
            'source': result['source'],
            'confidence': result['confidence'],
            'processing_time': round(processing_time, 3),
            'status': 'success',
            'metadata': {
                'similarity_score': result.get('similarity_score', 0),
                'reranker_score': result.get('reranker_score', 0),
                'intent_score': result.get('intent_score', 0),
                'top_matches': result.get('top_matches', [])
            }
        }
        
        logger.info(f"✅ Query processed successfully. Source: {result['source']}, Confidence: {result['confidence']:.3f}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@main_bp.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    try:
        cache_engine = current_app.cache_engine
        stats = cache_engine.get_stats()
        
        return jsonify({
            'stats': stats,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting cache stats: {e}")
        return jsonify({
            'error': f'Failed to get cache stats: {str(e)}',
            'status': 'error'
        }), 500

@main_bp.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the semantic cache"""
    try:
        cache_engine = current_app.cache_engine
        cache_engine.clear_cache()
        
        logger.info("🗑️ Cache cleared successfully")
        
        return jsonify({
            'message': 'Cache cleared successfully',
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")
        return jsonify({
            'error': f'Failed to clear cache: {str(e)}',
            'status': 'error'
        }), 500