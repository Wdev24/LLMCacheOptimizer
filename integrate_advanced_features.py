"""
Integration script to add advanced features to existing cache system
"""

import os
import shutil
from datetime import datetime

def backup_current_system():
    """Backup current system before enhancement"""
    backup_dir = f"backup_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        "core/cache_engine.py",
        "routes/main.py"
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{os.path.basename(file)}")
    
    print(f"✅ System backed up to {backup_dir}/")
    return backup_dir

def create_enhanced_route():
    """Create new enhanced API route"""
    
    enhanced_route = '''
# Add this to your routes/main.py file

@main_bp.route('/api/query/enhanced', methods=['POST'])
def process_query_enhanced():
    """
    Process user query through enhanced multi-stage semantic cache system
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
        
        logger.info(f"🔍 Processing enhanced query: '{query}'")
        
        # Get enhanced cache engine from app context
        if hasattr(current_app, 'enhanced_cache_engine'):
            enhanced_engine = current_app.enhanced_cache_engine
            result = enhanced_engine.process_query_enhanced(query)
        else:
            # Fallback to regular cache engine
            cache_engine = current_app.cache_engine
            result = cache_engine.process_query(query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result['api_processing_time'] = round(processing_time, 3)
        
        logger.info(f"✅ Enhanced query processed. Source: {result.get('source')}")
        
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
'''
    
    print("📝 Enhanced route code generated")
    print("Add this to your routes/main.py file:")
    print(enhanced_route)

def create_enhanced_frontend():
    """Create enhanced frontend with stage timing display"""
    
    enhanced_html = '''
<!-- Add this section to your templates/index.html after the existing response section -->

<!-- Enhanced Response Section -->
<div class="enhanced-response-section" id="enhancedResponseSection" style="display: none;">
    <div class="response-header">
        <div class="source-badge" id="enhancedSourceBadge">ENHANCED CACHE</div>
        <div class="confidence-meter">
            <span>Confidence:</span>
            <div class="confidence-bar">
                <div class="confidence-fill" id="enhancedConfidenceFill"></div>
            </div>
            <span id="enhancedConfidenceText">0%</span>
        </div>
    </div>

    <div class="response-text" id="enhancedResponseText"></div>

    <!-- Stage Timing Display -->
    <div class="stage-timings" id="stageTimings" style="margin-top: 20px;">
        <h4 style="margin-bottom: 10px;">🔍 Pipeline Stage Timings:</h4>
        <div class="timing-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
            <!-- Timing items will be populated dynamically -->
        </div>
    </div>

    <div class="query-decomposition" id="queryDecomposition" style="margin-top: 20px;">
        <h4 style="margin-bottom: 10px;">🧠 Query Analysis:</h4>
        <div class="decomp-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <!-- Decomposition items will be populated dynamically -->
        </div>
    </div>
</div>

<style>
.enhanced-response-section {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #3498db;
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 30px;
}

.stage-timings {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e8ed;
}

.timing-item {
    background: #f8f9fa;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    border-left: 4px solid #3498db;
}

.timing-label {
    font-size: 12px;
    color: #7f8c8d;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.timing-value {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
}

.query-decomposition {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e8ed;
}

.decomp-item {
    background: #f1f3f4;
    padding: 12px;
    border-radius: 8px;
}

.decomp-label {
    font-size: 12px;
    color: #7f8c8d;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.decomp-value {
    font-size: 14px;
    color: #2c3e50;
}
</style>

<script>
// Add this JavaScript function to handle enhanced responses

function displayEnhancedResponse(data) {
    // Show enhanced response section
    document.getElementById('enhancedResponseSection').style.display = 'block';
    
    // Update source badge
    const sourceBadge = document.getElementById('enhancedSourceBadge');
    sourceBadge.textContent = (data.source || 'unknown').toUpperCase();
    sourceBadge.className = `source-badge source-${data.source || 'error'}`;
    
    // Update confidence
    const confidence = Math.round((data.confidence || 0) * 100);
    document.getElementById('enhancedConfidenceFill').style.width = `${confidence}%`;
    document.getElementById('enhancedConfidenceText').textContent = `${confidence}%`;
    
    // Update response text
    document.getElementById('enhancedResponseText').textContent = data.response || 'No response';
    
    // Display stage timings
    displayStageTimings(data.stage_timings || {});
    
    // Display query decomposition
    displayQueryDecomposition(data.query_decomposition || {});
}

function displayStageTimings(timings) {
    const container = document.querySelector('.timing-grid');
    container.innerHTML = '';
    
    const stageLabels = {
        'decomposition': 'Query Analysis',
        'retrieval': 'Multi-Vector Search',
        'reranking': 'Ensemble Reranking',
        'calibration': 'Confidence Calibration',
        'llm_generation': 'LLM Generation',
        'llm_fallback': 'LLM Fallback'
    };
    
    for (const [stage, time] of Object.entries(timings)) {
        const timingItem = document.createElement('div');
        timingItem.className = 'timing-item';
        timingItem.innerHTML = `
            <div class="timing-label">${stageLabels[stage] || stage}</div>
            <div class="timing-value">${(time * 1000).toFixed(1)}ms</div>
        `;
        container.appendChild(timingItem);
    }
}

function displayQueryDecomposition(decomp) {
    const container = document.querySelector('.decomp-grid');
    container.innerHTML = '';
    
    const items = [
        { label: 'Query Type', value: decomp.query_type || 'unknown' },
        { label: 'Complexity', value: decomp.complexity_score ? `${(decomp.complexity_score * 100).toFixed(1)}%` : '0%' },
        { label: 'Atomic Intents', value: decomp.atomic_intents ? decomp.atomic_intents.length : 0 },
        { label: 'Alternates', value: decomp.alternate_phrasings ? decomp.alternate_phrasings.length : 0 }
    ];
    
    items.forEach(item => {
        const decompItem = document.createElement('div');
        decompItem.className = 'decomp-item';
        decompItem.innerHTML = `
            <div class="decomp-label">${item.label}</div>
            <div class="decomp-value">${item.value}</div>
        `;
        container.appendChild(decompItem);
    });
}

// Add enhanced query button handler
function addEnhancedQueryButton() {
    const queryForm = document.getElementById('queryForm');
    const enhancedBtn = document.createElement('button');
    enhancedBtn.type = 'button';
    enhancedBtn.className = 'submit-btn';
    enhancedBtn.innerHTML = '<span>🚀</span> Enhanced Query';
    enhancedBtn.onclick = handleEnhancedQuery;
    
    queryForm.appendChild(enhancedBtn);
}

async function handleEnhancedQuery() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) return;

    showLoading(true);
    hideError();
    
    try {
        const response = await fetch('/api/query/enhanced', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayEnhancedResponse(data);
        } else {
            showError(data.error || 'Unknown error occurred');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Initialize enhanced features when page loads
document.addEventListener('DOMContentLoaded', function() {
    addEnhancedQueryButton();
});
</script>
'''
    
    print("🎨 Enhanced frontend code generated")
    print("Add this to your templates/index.html file")

def create_app_integration():
    """Create app.py integration code"""
    
    app_integration = '''
# Add this to your app.py file after initializing the regular cache_engine

def initialize_enhanced_cache_system():
    """Initialize the enhanced semantic cache system"""
    try:
        from enhanced_multi_stage_cache import EnhancedSemanticCacheEngine
        from core.cache_engine import SemanticCacheEngine
        
        # Get existing components
        regular_engine = cache_engine
        
        # Create enhanced engine
        enhanced_engine = EnhancedSemanticCacheEngine(
            embedding_model=regular_engine.embedding_model,
            reranker_model=regular_engine.reranker,
            faiss_manager=regular_engine.faiss_manager,
            llm_client=regular_engine.llm_client,
            entity_extractor=getattr(regular_engine, 'entity_extractor', None)
        )
        
        logging.info("✅ Enhanced Semantic Cache System initialized")
        return enhanced_engine
        
    except ImportError as e:
        logging.warning(f"⚠️ Enhanced cache not available: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Failed to initialize enhanced cache: {e}")
        return None

# Add this in your main section after creating the regular cache_engine:
if __name__ == '__main__':
    # Initialize regular cache system
    cache_engine = initialize_cache_system()
    
    # Initialize enhanced cache system
    enhanced_cache_engine = initialize_enhanced_cache_system()
    
    # Create Flask app
    app = create_app()
    
    # Store both engines in app context
    app.cache_engine = cache_engine
    if enhanced_cache_engine:
        app.enhanced_cache_engine = enhanced_cache_engine
        print("🚀 Enhanced cache engine loaded!")
    else:
        print("⚠️ Running with regular cache engine only")
    
    print("🚀 Starting Hybrid Semantic Cache System...")
    print("📍 Access the system at: http://localhost:5000")
    print("🔬 Enhanced features available at /api/query/enhanced")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
    
    print("🔧 App integration code generated")
    print("Add this to your app.py file")

def create_advanced_tester():
    """Create tester for advanced features"""
    
    tester_code = '''
#!/usr/bin/env python3
"""
Advanced Feature Tester
Tests the enhanced multi-stage pipeline
"""

import requests
import json
import time

def test_enhanced_features():
    """Test enhanced cache features"""
    
    test_queries = [
        # Query decomposition tests
        "What is machine learning and how does it work?",
        "Define AI and give examples plus applications",
        "Tell me about neural networks and deep learning",
        
        # Complex reasoning tests
        "How does reinforcement learning differ from supervised learning?",
        "What are the advantages and disadvantages of neural networks?",
        "Explain the relationship between AI, ML, and deep learning",
        
        # Context-aware tests
        "Call Ramesh Kumar for the meeting",
        "Set alarm for 10:50 AM tomorrow",
        "Book a flight to Paris and hotel for 3 nights",
    ]
    
    print("🧪 Testing Enhanced Multi-Stage Pipeline")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n[{i}/{len(test_queries)}] Testing: '{query}'")
        
        # Test regular API
        regular_result = test_regular_api(query)
        
        # Test enhanced API
        enhanced_result = test_enhanced_api(query)
        
        # Compare results
        compare_results(query, regular_result, enhanced_result)
        
        time.sleep(1)  # Brief pause

def test_regular_api(query):
    """Test regular API"""
    try:
        response = requests.post(
            "http://localhost:5000/api/query",
            json={"query": query},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

def test_enhanced_api(query):
    """Test enhanced API"""
    try:
        response = requests.post(
            "http://localhost:5000/api/query/enhanced",
            json={"query": query},
            timeout=20
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

def compare_results(query, regular, enhanced):
    """Compare regular vs enhanced results"""
    
    print(f"  📊 Results Comparison:")
    
    # Regular API
    if "error" not in regular:
        reg_source = regular.get("source", "unknown")
        reg_confidence = regular.get("confidence", 0)
        reg_time = regular.get("processing_time", 0)
        print(f"    Regular:  {reg_source} | {reg_confidence:.3f} | {reg_time:.3f}s")
    else:
        print(f"    Regular:  ERROR - {regular['error']}")
    
    # Enhanced API
    if "error" not in enhanced:
        enh_source = enhanced.get("source", "unknown")
        enh_confidence = enhanced.get("confidence", 0)
        enh_time = enhanced.get("total_time", 0)
        
        # Stage timings
        stage_timings = enhanced.get("stage_timings", {})
        timing_summary = " | ".join([f"{k}:{v*1000:.1f}ms" for k, v in stage_timings.items()])
        
        print(f"    Enhanced: {enh_source} | {enh_confidence:.3f} | {enh_time:.3f}s")
        if timing_summary:
            print(f"    Stages:   {timing_summary}")
        
        # Query decomposition
        decomp = enhanced.get("query_decomposition", {})
        if decomp:
            complexity = decomp.get("complexity_score", 0)
            query_type = decomp.get("query_type", "unknown")
            intents = len(decomp.get("atomic_intents", []))
            print(f"    Analysis: {query_type} | complexity:{complexity:.2f} | intents:{intents}")
            
    else:
        print(f"    Enhanced: ERROR - {enhanced['error']}")

if __name__ == "__main__":
    test_enhanced_features()
'''
    
    with open("test_enhanced_features.py", "w") as f:
        f.write(tester_code)
    
    print("✅ Created test_enhanced_features.py")

def main():
    """Main integration process"""
    print("🚀 Advanced Multi-Stage Cache Integration")
    print("=" * 60)
    
    # Step 1: Backup current system
    backup_dir = backup_current_system()
    
    # Step 2: Check if enhanced_multi_stage_cache.py exists
    if not os.path.exists("enhanced_multi_stage_cache.py"):
        print("❌ enhanced_multi_stage_cache.py not found!")
        print("Please save the enhanced_multi_stage_cache.py code from the artifact above.")
        return False
    
    print("✅ Enhanced cache module found")
    
    # Step 3: Generate integration code
    print("\n📝 Generating integration code...")
    create_enhanced_route()
    create_enhanced_frontend()
    create_app_integration()
    create_advanced_tester()
    
    print(f"\n🎯 Integration Steps:")
    print(f"1. ✅ System backed up to {backup_dir}")
    print(f"2. 📝 Add the enhanced route code to routes/main.py")
    print(f"3. 🎨 Add the enhanced frontend code to templates/index.html")
    print(f"4. 🔧 Add the app integration code to app.py")
    print(f"5. 🔄 Restart your Flask app: python app.py")
    print(f"6. 🧪 Test with: python test_enhanced_features.py")
    
    print(f"\n🚀 Expected Benefits:")
    print(f"   📈 Improved accuracy through multi-stage validation")
    print(f"   🧠 Query decomposition for complex queries")
    print(f"   📊 Detailed pipeline timing analysis")
    print(f"   🎯 Context-aware LLM fallback")
    print(f"   🔧 Bayesian confidence calibration")
    
    return True

if __name__ == "__main__":
    main()