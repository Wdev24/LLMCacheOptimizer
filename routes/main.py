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

# ============================================================================
# 🚀 ENHANCED ROUTE - ADD THIS SECTION BELOW
# ============================================================================

@main_bp.route('/api/query/enhanced', methods=['POST'])
def process_query_enhanced():
    """
    Process user query through enhanced multi-stage semantic cache system
    Returns: JSON response with enhanced features and pipeline details
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
        
        logger.info(f"🚀 Processing enhanced query: '{query}'")
        
        # Get enhanced cache engine from app context
        if hasattr(current_app, 'enhanced_cache_engine') and current_app.enhanced_cache_engine:
            enhanced_engine = current_app.enhanced_cache_engine
            result = enhanced_engine.process_query_enhanced(query)
        else:
            # Fallback to regular cache engine if enhanced not available
            logger.warning("Enhanced cache engine not available, falling back to regular")
            cache_engine = current_app.cache_engine
            result = cache_engine.process_query(query)
            # Add enhanced-style metadata
            result['stage_timings'] = {'regular_processing': result.get('processing_time', 0)}
            result['query_decomposition'] = {
                'original_query': query,
                'query_type': 'fallback',
                'complexity_score': 0.5
            }
        
        # Calculate API processing time
        api_processing_time = time.time() - start_time
        result['api_processing_time'] = round(api_processing_time, 3)
        
        logger.info(f"✅ Enhanced query processed. Source: {result.get('source')}, Confidence: {result.get('confidence', 0):.3f}")
        
        return jsonify({
            'status': 'success',
            **result
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing enhanced query: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500