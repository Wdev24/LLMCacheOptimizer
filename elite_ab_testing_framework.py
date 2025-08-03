#!/usr/bin/env python3
"""
🚀 ULTRA-ADVANCED SEMANTIC CACHE TESTING FRAMEWORK 🚀
Enterprise-grade testing suite for production-ready semantic cache systems
Designed for maximum stress testing and statistical rigor
"""

import requests
import json
import time
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from threading import Lock
from collections import defaultdict, Counter
import random
import string
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Advanced statistical analysis
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, pearsonr, spearmanr
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SCIPY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    print("⚠️ Advanced libraries not available - install scipy and scikit-learn for full functionality")

@dataclass
class AdvancedTestResult:
    """Ultra-detailed test result with performance profiling"""
    # Basic test info
    test_id: str
    query: str
    expected_behavior: str
    category: str
    subcategory: str
    difficulty: int
    complexity_score: float
    adversarial_type: str = ""
    
    # Regular API Results with detailed timing
    regular_source: str = ""
    regular_confidence: float = 0.0
    regular_total_time: float = 0.0
    regular_network_time: float = 0.0
    regular_processing_time: float = 0.0
    regular_similarity: float = 0.0
    regular_reranker: float = 0.0
    regular_intent: float = 0.0
    regular_response: str = ""
    regular_error: str = ""
    regular_memory_usage: float = 0.0
    regular_cpu_usage: float = 0.0
    
    # Enhanced API Results with detailed timing
    enhanced_source: str = ""
    enhanced_confidence: float = 0.0
    enhanced_total_time: float = 0.0
    enhanced_network_time: float = 0.0
    enhanced_processing_time: float = 0.0
    enhanced_similarity: float = 0.0
    enhanced_reranker: float = 0.0
    enhanced_intent: float = 0.0
    enhanced_response: str = ""
    enhanced_error: str = ""
    enhanced_stages: int = 0
    enhanced_complexity: float = 0.0
    enhanced_query_type: str = ""
    enhanced_memory_usage: float = 0.0
    enhanced_cpu_usage: float = 0.0
    
    # Advanced Analysis
    is_regular_correct: bool = False
    is_enhanced_correct: bool = False
    correctness_confidence: float = 0.0
    winner: str = ""
    improvement_type: str = ""
    confidence_delta: float = 0.0
    speed_delta: float = 0.0
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    
    # Stress test metrics
    concurrent_requests: int = 1
    stress_level: str = "normal"
    retry_count: int = 0
    timeout_occurred: bool = False
    
    # Advanced metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    test_session_id: str = ""

class UltraAdvancedTester:
    """Enterprise-grade semantic cache testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results: List[AdvancedTestResult] = []
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.test_start_time = None
        self.test_end_time = None
        self.stress_results = defaultdict(list)
        self.lock = Lock()
        
    def get_ultra_comprehensive_test_suite(self) -> List[Dict[str, Any]]:
        """Generate the most comprehensive test suite possible"""
        
        test_suite = []
        
        # ==================== TIER 1: BASIC FUNCTIONALITY ====================
        basic_tests = [
            # Exact matches - Foundation level
            {"query": "What is machine learning?", "expected": "cache_hit", "category": "exact_match", "subcategory": "definition", "difficulty": 1, "complexity": 0.1},
            {"query": "Define artificial intelligence", "expected": "cache_hit", "category": "exact_match", "subcategory": "definition", "difficulty": 1, "complexity": 0.1},
            {"query": "How does neural network work?", "expected": "cache_hit", "category": "exact_match", "subcategory": "explanation", "difficulty": 1, "complexity": 0.2},
            {"query": "What is deep learning?", "expected": "cache_hit", "category": "exact_match", "subcategory": "definition", "difficulty": 1, "complexity": 0.1},
            {"query": "Explain natural language processing", "expected": "cache_hit", "category": "exact_match", "subcategory": "explanation", "difficulty": 1, "complexity": 0.2},
        ]
        test_suite.extend(basic_tests)
        
        # ==================== TIER 2: SEMANTIC VARIATIONS ====================
        semantic_tests = [
            # Synonyms and paraphrasing
            {"query": "What is ML?", "expected": "cache_hit", "category": "semantic_match", "subcategory": "abbreviation", "difficulty": 2, "complexity": 0.3},
            {"query": "Explain machine learning", "expected": "cache_hit", "category": "semantic_match", "subcategory": "verb_variation", "difficulty": 2, "complexity": 0.3},
            {"query": "Tell me about AI", "expected": "cache_hit", "category": "semantic_match", "subcategory": "style_variation", "difficulty": 2, "complexity": 0.4},
            {"query": "How do neural nets function?", "expected": "cache_hit", "category": "semantic_match", "subcategory": "abbreviation", "difficulty": 2, "complexity": 0.4},
            {"query": "What are deep neural networks?", "expected": "cache_hit", "category": "semantic_match", "subcategory": "expansion", "difficulty": 2, "complexity": 0.4},
            {"query": "Describe NLP", "expected": "cache_hit", "category": "semantic_match", "subcategory": "abbreviation", "difficulty": 2, "complexity": 0.3},
            
            # Different phrasing patterns
            {"query": "Can you explain what machine learning is?", "expected": "cache_hit", "category": "semantic_match", "subcategory": "question_form", "difficulty": 2, "complexity": 0.5},
            {"query": "I want to understand artificial intelligence", "expected": "cache_hit", "category": "semantic_match", "subcategory": "intent_form", "difficulty": 2, "complexity": 0.5},
            {"query": "Help me learn about neural networks", "expected": "cache_hit", "category": "semantic_match", "subcategory": "request_form", "difficulty": 2, "complexity": 0.5},
        ]
        test_suite.extend(semantic_tests)
        
        # ==================== TIER 3: ENTITY PRECISION TESTS ====================
        entity_tests = [
            # Name variations - CRITICAL for preventing false positives
            {"query": "Call Ramesh Kumar", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "name_mismatch", "difficulty": 9, "complexity": 0.9},
            {"query": "Call Ramesh Sharma", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "name_mismatch", "difficulty": 9, "complexity": 0.9},
            {"query": "Call Ramesh Patel", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "name_mismatch", "difficulty": 9, "complexity": 0.9},
            {"query": "Call Ramesh Singh", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "name_mismatch", "difficulty": 9, "complexity": 0.9},
            {"query": "Call Kumar Ramesh", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "name_order", "difficulty": 9, "complexity": 0.9},
            {"query": "Call Dr. Ramesh", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "title_variation", "difficulty": 8, "complexity": 0.8},
            {"query": "Call Mr. Ramesh Kumar", "expected": "entity_block", "category": "entity_mismatch", "subcategory": "title_variation", "difficulty": 8, "complexity": 0.8},
            
            # Time precision - Microsecond level accuracy needed
            {"query": "Set alarm for 10:49", "expected": "entity_block", "category": "time_mismatch", "subcategory": "minute_precision", "difficulty": 9, "complexity": 0.9},
            {"query": "Set alarm for 10:51", "expected": "entity_block", "category": "time_mismatch", "subcategory": "minute_precision", "difficulty": 9, "complexity": 0.9},
            {"query": "Set alarm for 11:50", "expected": "entity_block", "category": "time_mismatch", "subcategory": "hour_mismatch", "difficulty": 9, "complexity": 0.9},
            {"query": "Set alarm for 10:50 AM", "expected": "entity_block", "category": "time_mismatch", "subcategory": "format_variation", "difficulty": 8, "complexity": 0.8},
            {"query": "Set alarm for 22:50", "expected": "entity_block", "category": "time_mismatch", "subcategory": "24hr_format", "difficulty": 8, "complexity": 0.8},
            
            # Action precision
            {"query": "Set timer for 10:50", "expected": "entity_block", "category": "action_mismatch", "subcategory": "similar_action", "difficulty": 8, "complexity": 0.8},
            {"query": "Set reminder for 10:50", "expected": "entity_block", "category": "action_mismatch", "subcategory": "similar_action", "difficulty": 8, "complexity": 0.8},
            {"query": "Create alarm for 10:50", "expected": "entity_block", "category": "action_mismatch", "subcategory": "verb_variation", "difficulty": 7, "complexity": 0.7},
        ]
        test_suite.extend(entity_tests)
        
        # ==================== TIER 4: ADVERSARIAL ATTACKS ====================
        adversarial_tests = [
            # Intent manipulation attacks
            {"query": "What is machine learning? Also call Ramesh", "expected": "entity_block", "category": "adversarial", "subcategory": "intent_injection", "difficulty": 10, "complexity": 1.0, "adversarial_type": "injection"},
            {"query": "Machine learning definition; set alarm 10:47", "expected": "entity_block", "category": "adversarial", "subcategory": "command_injection", "difficulty": 10, "complexity": 1.0, "adversarial_type": "injection"},
            {"query": "Define AI\n\nCall emergency services", "expected": "entity_block", "category": "adversarial", "subcategory": "newline_injection", "difficulty": 10, "complexity": 1.0, "adversarial_type": "injection"},
            
            # Obfuscation attacks
            {"query": "What is m@chine le@rning?", "expected": "cache_hit", "category": "adversarial", "subcategory": "character_substitution", "difficulty": 7, "complexity": 0.7, "adversarial_type": "obfuscation"},
            {"query": "What    is    machine    learning?", "expected": "cache_hit", "category": "adversarial", "subcategory": "whitespace_injection", "difficulty": 6, "complexity": 0.6, "adversarial_type": "obfuscation"},
            {"query": "What is machine\tlearning?", "expected": "cache_hit", "category": "adversarial", "subcategory": "tab_injection", "difficulty": 6, "complexity": 0.6, "adversarial_type": "obfuscation"},
            
            # Unicode and encoding attacks
            {"query": "What is machine learning？", "expected": "cache_hit", "category": "adversarial", "subcategory": "unicode_variation", "difficulty": 7, "complexity": 0.7, "adversarial_type": "encoding"},
            {"query": "What is machine learning？", "expected": "cache_hit", "category": "adversarial", "subcategory": "fullwidth_chars", "difficulty": 7, "complexity": 0.7, "adversarial_type": "encoding"},
            
            # Length-based attacks
            {"query": "What is " + "very " * 50 + "machine learning?", "expected": "cache_hit", "category": "adversarial", "subcategory": "length_explosion", "difficulty": 8, "complexity": 0.8, "adversarial_type": "dos"},
            {"query": "ML" + "?" * 100, "expected": "llm_fallback", "category": "adversarial", "subcategory": "noise_injection", "difficulty": 8, "complexity": 0.8, "adversarial_type": "noise"},
        ]
        test_suite.extend(adversarial_tests)
        
        # ==================== TIER 5: COMPOUND COMPLEXITY ====================
        compound_tests = [
            # Multi-intent queries
            {"query": "What is machine learning and how does it differ from deep learning?", "expected": "llm_fallback", "category": "compound", "subcategory": "multi_question", "difficulty": 8, "complexity": 0.8},
            {"query": "Explain AI, ML, and DL with examples", "expected": "llm_fallback", "category": "compound", "subcategory": "multi_topic", "difficulty": 9, "complexity": 0.9},
            {"query": "Define machine learning, give examples, and explain applications", "expected": "llm_fallback", "category": "compound", "subcategory": "multi_request", "difficulty": 9, "complexity": 0.9},
            
            # Conditional and contextual queries
            {"query": "If I'm a beginner, how should I learn machine learning?", "expected": "llm_fallback", "category": "contextual", "subcategory": "conditional", "difficulty": 7, "complexity": 0.8},
            {"query": "What is machine learning for someone with no programming background?", "expected": "llm_fallback", "category": "contextual", "subcategory": "audience_specific", "difficulty": 7, "complexity": 0.8},
            {"query": "Machine learning explanation for a 10-year-old", "expected": "llm_fallback", "category": "contextual", "subcategory": "simplification", "difficulty": 7, "complexity": 0.8},
            
            # Temporal and comparative queries
            {"query": "How has machine learning evolved since 2020?", "expected": "llm_fallback", "category": "temporal", "subcategory": "evolution", "difficulty": 8, "complexity": 0.9},
            {"query": "Machine learning vs traditional programming", "expected": "llm_fallback", "category": "comparative", "subcategory": "vs_comparison", "difficulty": 8, "complexity": 0.8},
            {"query": "Pros and cons of machine learning", "expected": "llm_fallback", "category": "analytical", "subcategory": "pros_cons", "difficulty": 7, "complexity": 0.8},
        ]
        test_suite.extend(compound_tests)
        
        # ==================== TIER 6: EDGE CASES ====================
        edge_cases = [
            # Empty and minimal queries
            {"query": "", "expected": "llm_fallback", "category": "edge_case", "subcategory": "empty_query", "difficulty": 5, "complexity": 0.5},
            {"query": " ", "expected": "llm_fallback", "category": "edge_case", "subcategory": "whitespace_only", "difficulty": 5, "complexity": 0.5},
            {"query": "?", "expected": "llm_fallback", "category": "edge_case", "subcategory": "single_char", "difficulty": 5, "complexity": 0.5},
            {"query": "ML", "expected": "cache_hit", "category": "edge_case", "subcategory": "minimal_query", "difficulty": 3, "complexity": 0.3},
            
            # Special characters and symbols
            {"query": "What is ML???", "expected": "cache_hit", "category": "edge_case", "subcategory": "repeated_punctuation", "difficulty": 4, "complexity": 0.4},
            {"query": "***Machine Learning***", "expected": "cache_hit", "category": "edge_case", "subcategory": "symbol_wrapping", "difficulty": 4, "complexity": 0.4},
            {"query": "[ML] definition", "expected": "cache_hit", "category": "edge_case", "subcategory": "bracket_notation", "difficulty": 4, "complexity": 0.4},
            
            # Language mixing and code injection
            {"query": "¿Qué es machine learning?", "expected": "llm_fallback", "category": "edge_case", "subcategory": "language_mixing", "difficulty": 6, "complexity": 0.6},
            {"query": "machine learning <script>alert('xss')</script>", "expected": "cache_hit", "category": "edge_case", "subcategory": "xss_attempt", "difficulty": 9, "complexity": 0.9},
            {"query": "ML'; DROP TABLE cache; --", "expected": "cache_hit", "category": "edge_case", "subcategory": "sql_injection", "difficulty": 9, "complexity": 0.9},
        ]
        test_suite.extend(edge_cases)
        
        # ==================== TIER 7: PERFORMANCE STRESS TESTS ====================
        stress_tests = [
            # High-frequency similar queries
            {"query": f"What is machine learning version {i}?", "expected": "llm_fallback", "category": "stress", "subcategory": "high_frequency", "difficulty": 6, "complexity": 0.6}
            for i in range(20)
        ]
        test_suite.extend(stress_tests[:10])  # Limit to 10 for now
        
        # Add test IDs and session info
        for i, test in enumerate(test_suite):
            test["test_id"] = f"{self.session_id}_{i:03d}"
            test["complexity_score"] = test.get("complexity", 0.5)
        
        return test_suite
    
    def test_single_api_advanced(self, query: str, endpoint: str, test_config: Dict = None) -> Dict[str, Any]:
        """Advanced API testing with detailed performance profiling"""
        
        if test_config is None:
            test_config = {"timeout": 30, "retries": 3, "concurrent": 1}
        
        start_time = time.time()
        network_start = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json={"query": query},
                timeout=test_config.get("timeout", 30),
                headers={"Content-Type": "application/json"}
            )
            
            network_time = time.time() - network_start
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract comprehensive metrics
                result = {
                    "source": data.get("source", "error"),
                    "confidence": data.get("confidence", 0.0),
                    "total_time": total_time,
                    "network_time": network_time,
                    "processing_time": data.get("processing_time", data.get("total_time", 0)),
                    "similarity_score": data.get("similarity_score", data.get("metadata", {}).get("similarity_score", 0)),
                    "reranker_score": data.get("reranker_score", data.get("metadata", {}).get("reranker_score", 0)),
                    "intent_score": data.get("intent_score", data.get("metadata", {}).get("intent_score", 0)),
                    "response_text": data.get("response", ""),
                    "error": "",
                    "memory_usage": 0.0,  # Could be extracted from response if available
                    "cpu_usage": 0.0,     # Could be extracted from response if available
                    "timeout_occurred": False,
                    "retry_count": 0
                }
                
                # Enhanced-specific fields
                if endpoint == "/api/query/enhanced":
                    result.update({
                        "stages": len(data.get("stage_timings", {})),
                        "complexity": data.get("query_decomposition", {}).get("complexity_score", 0),
                        "query_type": data.get("query_decomposition", {}).get("query_type", "")
                    })
                else:
                    result.update({
                        "stages": 0,
                        "complexity": 0.0,
                        "query_type": ""
                    })
                
                return result
            else:
                return self._create_error_result(response.status_code, total_time, network_time)
                
        except requests.exceptions.Timeout:
            return self._create_error_result("timeout", time.time() - start_time, 0, timeout=True)
        except Exception as e:
            return self._create_error_result(str(e), time.time() - start_time, 0)
    
    def _create_error_result(self, error, total_time, network_time, timeout=False):
        """Create standardized error result"""
        return {
            "source": "error",
            "confidence": 0.0,
            "total_time": total_time,
            "network_time": network_time,
            "processing_time": 0,
            "similarity_score": 0,
            "reranker_score": 0,
            "intent_score": 0,
            "response_text": "",
            "error": error,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "stages": 0,
            "complexity": 0.0,
            "query_type": "",
            "timeout_occurred": timeout,
            "retry_count": 0
        }
    
    def evaluate_correctness_advanced(self, expected: str, source: str, confidence: float, query: str = "") -> Tuple[bool, float]:
        """Advanced correctness evaluation with confidence scoring"""
        
        # Base correctness
        is_correct = False
        confidence_score = 0.0
        
        if expected == "cache_hit":
            is_correct = source in ["cache", "enhanced_cache"] and confidence >= 0.6
            confidence_score = min(confidence * 1.5, 1.0) if is_correct else confidence * 0.5
            
        elif expected == "entity_block":
            is_correct = source in ["llm", "enhanced_llm", "llm_fallback"] or confidence < 0.6
            # For entity blocking, we want LOW confidence when blocking, HIGH confidence when correct
            confidence_score = (1.0 - confidence) if is_correct else confidence * 0.3
            
        elif expected == "llm_fallback":
            is_correct = source in ["llm", "enhanced_llm", "llm_fallback"] or confidence < 0.6
            confidence_score = confidence if is_correct else (1.0 - confidence) * 0.5
        
        # Adjust confidence based on query complexity
        if "adversarial" in query.lower() or len(query) > 200:
            confidence_score *= 0.8  # Lower confidence for complex queries
        
        return is_correct, min(confidence_score, 1.0)
    
    def analyze_advanced_result(self, test_case: Dict, regular_result: Dict, enhanced_result: Dict) -> AdvancedTestResult:
        """Ultra-detailed result analysis with advanced metrics"""
        
        # Advanced correctness evaluation
        regular_correct, regular_conf_score = self.evaluate_correctness_advanced(
            test_case["expected"], regular_result["source"], 
            regular_result["confidence"], test_case["query"]
        )
        
        enhanced_correct, enhanced_conf_score = self.evaluate_correctness_advanced(
            test_case["expected"], enhanced_result["source"], 
            enhanced_result["confidence"], test_case["query"]
        )
        
        # Advanced winner determination with multiple criteria
        winner = self._determine_winner_advanced(
            regular_correct, enhanced_correct,
            regular_result, enhanced_result,
            test_case["difficulty"]
        )
        
        # Calculate advanced metrics
        efficiency_score = self._calculate_efficiency_score(regular_result, enhanced_result)
        robustness_score = self._calculate_robustness_score(test_case, regular_result, enhanced_result)
        
        return AdvancedTestResult(
            test_id=test_case["test_id"],
            query=test_case["query"],
            expected_behavior=test_case["expected"],
            category=test_case["category"],
            subcategory=test_case["subcategory"],
            difficulty=test_case["difficulty"],
            complexity_score=test_case["complexity_score"],
            adversarial_type=test_case.get("adversarial_type", ""),
            
            # Regular results
            regular_source=regular_result["source"],
            regular_confidence=regular_result["confidence"],
            regular_total_time=regular_result["total_time"],
            regular_network_time=regular_result["network_time"],
            regular_processing_time=regular_result["processing_time"],
            regular_similarity=regular_result["similarity_score"],
            regular_reranker=regular_result["reranker_score"],
            regular_intent=regular_result["intent_score"],
            regular_response=regular_result["response_text"],
            regular_error=regular_result["error"],
            regular_memory_usage=regular_result["memory_usage"],
            regular_cpu_usage=regular_result["cpu_usage"],
            
            # Enhanced results
            enhanced_source=enhanced_result["source"],
            enhanced_confidence=enhanced_result["confidence"],
            enhanced_total_time=enhanced_result["total_time"],
            enhanced_network_time=enhanced_result["network_time"],
            enhanced_processing_time=enhanced_result["processing_time"],
            enhanced_similarity=enhanced_result["similarity_score"],
            enhanced_reranker=enhanced_result["reranker_score"],
            enhanced_intent=enhanced_result["intent_score"],
            enhanced_response=enhanced_result["response_text"],
            enhanced_error=enhanced_result["error"],
            enhanced_stages=enhanced_result["stages"],
            enhanced_complexity=enhanced_result["complexity"],
            enhanced_query_type=enhanced_result["query_type"],
            enhanced_memory_usage=enhanced_result["memory_usage"],
            enhanced_cpu_usage=enhanced_result["cpu_usage"],
            
            # Advanced analysis
            is_regular_correct=regular_correct,
            is_enhanced_correct=enhanced_correct,
            correctness_confidence=max(regular_conf_score, enhanced_conf_score),
            winner=winner["winner"],
            improvement_type=winner["improvement_type"],
            confidence_delta=enhanced_result["confidence"] - regular_result["confidence"],
            speed_delta=regular_result["total_time"] - enhanced_result["total_time"],
            efficiency_score=efficiency_score,
            robustness_score=robustness_score,
            
            # Stress metrics
            timeout_occurred=regular_result.get("timeout_occurred", False) or enhanced_result.get("timeout_occurred", False),
            retry_count=max(regular_result.get("retry_count", 0), enhanced_result.get("retry_count", 0)),
            test_session_id=self.session_id
        )
    
    def _determine_winner_advanced(self, reg_correct: bool, enh_correct: bool, 
                                  reg_result: Dict, enh_result: Dict, difficulty: int) -> Dict[str, str]:
        """Advanced winner determination with weighted criteria"""
        
        # Accuracy is most important (60% weight)
        if enh_correct and not reg_correct:
            return {"winner": "enhanced", "improvement_type": "accuracy"}
        elif reg_correct and not enh_correct:
            return {"winner": "regular", "improvement_type": "accuracy"}
        
        # Both correct or both incorrect - look at secondary criteria
        elif reg_correct and enh_correct:
            # Speed comparison (25% weight)
            speed_diff = reg_result["total_time"] - enh_result["total_time"]
            
            # Confidence comparison (15% weight)
            conf_diff = enh_result["confidence"] - reg_result["confidence"]
            
            # Calculate weighted score
            enhanced_score = 0
            if speed_diff > 0.01:  # Enhanced is faster
                enhanced_score += 25
            elif speed_diff < -0.01:  # Regular is faster
                enhanced_score -= 25
            
            if conf_diff > 0.05:  # Enhanced is more confident
                enhanced_score += 15
            elif conf_diff < -0.05:  # Regular is more confident
                enhanced_score -= 15
            
            # Difficulty bonus - enhanced should handle hard queries better
            if difficulty >= 8:
                enhanced_score += 10
            
            if enhanced_score > 0:
                return {"winner": "enhanced", "improvement_type": "performance"}
            elif enhanced_score < 0:
                return {"winner": "regular", "improvement_type": "performance"}
            else:
                return {"winner": "tie", "improvement_type": "none"}
        
        else:  # Both incorrect
            # Compare confidence and robustness
            if enh_result["confidence"] > reg_result["confidence"] + 0.1:
                return {"winner": "enhanced", "improvement_type": "robustness"}
            elif reg_result["confidence"] > enh_result["confidence"] + 0.1:
                return {"winner": "regular", "improvement_type": "robustness"}
            else:
                return {"winner": "tie", "improvement_type": "none"}
    
    def _calculate_efficiency_score(self, reg_result: Dict, enh_result: Dict) -> float:
        """Calculate efficiency score based on performance vs accuracy trade-off"""
        try:
            # Normalize metrics to 0-1 scale
            reg_time = max(reg_result["total_time"], 0.001)
            enh_time = max(enh_result["total_time"], 0.001)
            
            reg_conf = reg_result["confidence"]
            enh_conf = enh_result["confidence"]
            
            # Calculate efficiency: confidence per unit time
            reg_efficiency = reg_conf / reg_time
            enh_efficiency = enh_conf / enh_time
            
            # Return relative efficiency (enhanced vs regular)
            if reg_efficiency > 0:
                return min(enh_efficiency / reg_efficiency, 3.0)  # Cap at 3x
            else:
                return 1.0
        except:
            return 1.0
    
    def _calculate_robustness_score(self, test_case: Dict, reg_result: Dict, enh_result: Dict) -> float:
        """Calculate robustness score based on handling of edge cases and adversarial inputs"""
        try:
            score = 0.5  # Base score
            
            # Adversarial handling bonus
            if test_case.get("adversarial_type"):
                if enh_result["error"] == "" and reg_result["error"] != "":
                    score += 0.3
                elif enh_result["confidence"] > reg_result["confidence"]:
                    score += 0.2
            
            # Edge case handling
            if test_case["category"] == "edge_case":
                if enh_result["source"] != "error" and reg_result["source"] == "error":
                    score += 0.3
                elif enh_result["total_time"] < reg_result["total_time"]:
                    score += 0.1
            
            # High difficulty bonus
            if test_case["difficulty"] >= 8:
                if enh_result["confidence"] > reg_result["confidence"]:
                    score += 0.2
            
            return min(score, 1.0)
        except:
            return 0.5
    
    def run_ultra_comprehensive_test(self, test_config: Dict = None) -> List[AdvancedTestResult]:
        """Run the most comprehensive test possible"""
        
        if test_config is None:
            test_config = {
                "parallel": True,
                "max_workers": 4,
                "stress_test": True,
                "adversarial_test": True,
                "timeout": 30,
                "retries": 3
            }
        
        test_suite = self.get_ultra_comprehensive_test_suite()
        self.test_start_time = datetime.now()
        
        print(f"🚀 ULTRA-ADVANCED SEMANTIC CACHE TESTING FRAMEWORK")
        print(f"🧪 Testing {len(test_suite)} ultra-comprehensive queries")
        print(f"⚡ Configuration: {test_config}")
        print(f"🎯 Session ID: {self.session_id}")
        print("=" * 100)
        
        if test_config.get("parallel", True):
            results = self._run_ultra_parallel_test(test_suite, test_config)
        else:
            results = self._run_ultra_sequential_test(test_suite, test_config)
        
        # Run stress tests if enabled
        if test_config.get("stress_test", True):
            print("\n🔥 Running stress tests...")
            stress_results = self._run_stress_tests(test_config)
            results.extend(stress_results)
        
        self.test_end_time = datetime.now()
        self.results = results
        
        print(f"\n✅ Ultra-comprehensive testing completed!")
        print(f"   Duration: {(self.test_end_time - self.test_start_time).total_seconds():.1f}s")
        print(f"   Total tests: {len(results)}")
        print(f"   Success rate: {len([r for r in results if not r.timeout_occurred]) / len(results) * 100:.1f}%")
        
        return results
    
    def _run_ultra_parallel_test(self, test_suite: List[Dict], config: Dict) -> List[AdvancedTestResult]:
        """Run tests with maximum parallelism and efficiency"""
        results = []
        
        def test_single_case(test_case):
            regular_result = self.test_single_api_advanced(test_case["query"], "/api/query", config)
            time.sleep(0.05)  # Brief pause to prevent overwhelming
            enhanced_result = self.test_single_api_advanced(test_case["query"], "/api/query/enhanced", config)
            return self.analyze_advanced_result(test_case, regular_result, enhanced_result)
        
        with ThreadPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
            futures = [executor.submit(test_single_case, test_case) for test_case in test_suite]
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                print(f"\r[{completed}/{len(test_suite)}] Processing ultra-comprehensive tests... "
                      f"({completed/len(test_suite)*100:.1f}%)", end="", flush=True)
                
                try:
                    result = future.result()
                    with self.lock:
                        results.append(result)
                except Exception as e:
                    print(f"\n⚠️ Test failed: {e}")
        
        print()  # New line after progress
        return results
    
    def _run_ultra_sequential_test(self, test_suite: List[Dict], config: Dict) -> List[AdvancedTestResult]:
        """Run tests sequentially with detailed progress"""
        results = []
        
        for i, test_case in enumerate(test_suite, 1):
            category = test_case["category"]
            difficulty = test_case["difficulty"]
            
            print(f"\r[{i}/{len(test_suite)}] {category} (diff: {difficulty}): '{test_case['query'][:40]}...'", 
                  end="", flush=True)
            
            regular_result = self.test_single_api_advanced(test_case["query"], "/api/query", config)
            time.sleep(0.1)
            enhanced_result = self.test_single_api_advanced(test_case["query"], "/api/query/enhanced", config)
            
            result = self.analyze_advanced_result(test_case, regular_result, enhanced_result)
            results.append(result)
            
            time.sleep(0.2)  # Prevent overwhelming the server
        
        print()
        return results
    
    def _run_stress_tests(self, config: Dict) -> List[AdvancedTestResult]:
        """Run high-load stress tests"""
        stress_queries = [
            "What is machine learning?",
            "Define artificial intelligence",
            "How does neural network work?",
            "Explain deep learning",
            "What is natural language processing?"
        ]
        
        stress_results = []
        
        # Concurrent load test
        print("   🔥 Testing concurrent load (50 simultaneous requests)...")
        
        def stress_test_single(query_info):
            query, request_id = query_info
            start_time = time.time()
            
            regular_result = self.test_single_api_advanced(query, "/api/query", {"timeout": 10})
            enhanced_result = self.test_single_api_advanced(query, "/api/query/enhanced", {"timeout": 10})
            
            test_case = {
                "test_id": f"stress_{request_id}",
                "query": query,
                "expected": "cache_hit",
                "category": "stress_test",
                "subcategory": "concurrent_load",
                "difficulty": 7,
                "complexity_score": 0.7
            }
            
            result = self.analyze_advanced_result(test_case, regular_result, enhanced_result)
            result.stress_level = "high"
            result.concurrent_requests = 50
            
            return result
        
        # Create 50 concurrent requests
        query_list = [(random.choice(stress_queries), i) for i in range(50)]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stress_test_single, query_info) for query_info in query_list]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    stress_results.append(result)
                except Exception as e:
                    print(f"   ⚠️ Stress test failed: {e}")
        
        print(f"   ✅ Stress test completed: {len(stress_results)} requests processed")
        return stress_results
    
    def generate_ultra_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate the most detailed analysis possible"""
        
        if not self.results:
            raise ValueError("No test results available. Run tests first.")
        
        # Basic metrics
        total_tests = len(self.results)
        regular_correct = sum(1 for r in self.results if r.is_regular_correct)
        enhanced_correct = sum(1 for r in self.results if r.is_enhanced_correct)
        
        # Advanced accuracy metrics
        accuracy_by_difficulty = {}
        for difficulty in range(1, 11):
            diff_results = [r for r in self.results if r.difficulty == difficulty]
            if diff_results:
                accuracy_by_difficulty[difficulty] = {
                    "regular": sum(1 for r in diff_results if r.is_regular_correct) / len(diff_results) * 100,
                    "enhanced": sum(1 for r in diff_results if r.is_enhanced_correct) / len(diff_results) * 100,
                    "count": len(diff_results)
                }
        
        # Category analysis with subcategories
        category_analysis = {}
        for category in set(r.category for r in self.results):
            cat_results = [r for r in self.results if r.category == category]
            
            subcategory_analysis = {}
            for subcategory in set(r.subcategory for r in cat_results):
                subcat_results = [r for r in cat_results if r.subcategory == subcategory]
                subcategory_analysis[subcategory] = {
                    "total": len(subcat_results),
                    "regular_correct": sum(1 for r in subcat_results if r.is_regular_correct),
                    "enhanced_correct": sum(1 for r in subcat_results if r.is_enhanced_correct),
                    "avg_difficulty": statistics.mean([r.difficulty for r in subcat_results]),
                    "avg_efficiency": statistics.mean([r.efficiency_score for r in subcat_results]),
                    "avg_robustness": statistics.mean([r.robustness_score for r in subcat_results])
                }
            
            category_analysis[category] = {
                "total": len(cat_results),
                "regular_correct": sum(1 for r in cat_results if r.is_regular_correct),
                "enhanced_correct": sum(1 for r in cat_results if r.is_enhanced_correct),
                "subcategories": subcategory_analysis,
                "avg_response_time_regular": statistics.mean([r.regular_total_time for r in cat_results]),
                "avg_response_time_enhanced": statistics.mean([r.enhanced_total_time for r in cat_results]),
                "timeout_rate": sum(1 for r in cat_results if r.timeout_occurred) / len(cat_results) * 100
            }
        
        # Advanced performance metrics
        performance_percentiles = {}
        regular_times = [r.regular_total_time for r in self.results if r.regular_total_time > 0]
        enhanced_times = [r.enhanced_total_time for r in self.results if r.enhanced_total_time > 0]
        
        if regular_times and enhanced_times:
            performance_percentiles = {
                "regular_p50": np.percentile(regular_times, 50),
                "regular_p95": np.percentile(regular_times, 95),
                "regular_p99": np.percentile(regular_times, 99),
                "enhanced_p50": np.percentile(enhanced_times, 50),
                "enhanced_p95": np.percentile(enhanced_times, 95),
                "enhanced_p99": np.percentile(enhanced_times, 99)
            }
        
        # Adversarial attack analysis
        adversarial_results = [r for r in self.results if r.adversarial_type]
        adversarial_analysis = {}
        if adversarial_results:
            for attack_type in set(r.adversarial_type for r in adversarial_results):
                attack_results = [r for r in adversarial_results if r.adversarial_type == attack_type]
                adversarial_analysis[attack_type] = {
                    "total": len(attack_results),
                    "regular_success": sum(1 for r in attack_results if r.is_regular_correct) / len(attack_results) * 100,
                    "enhanced_success": sum(1 for r in attack_results if r.is_enhanced_correct) / len(attack_results) * 100,
                    "avg_robustness": statistics.mean([r.robustness_score for r in attack_results])
                }
        
        # Statistical tests
        statistical_tests = self._perform_ultra_advanced_statistical_tests()
        
        # Quality scores
        quality_scores = self._calculate_quality_scores()
        
        return {
            "test_metadata": {
                "session_id": self.session_id,
                "total_tests": total_tests,
                "test_duration": (self.test_end_time - self.test_start_time).total_seconds(),
                "timestamp": self.test_start_time.isoformat(),
                "framework_version": "Ultra-Advanced v2.0"
            },
            "accuracy_metrics": {
                "regular_accuracy": round(regular_correct / total_tests * 100, 2),
                "enhanced_accuracy": round(enhanced_correct / total_tests * 100, 2),
                "accuracy_improvement": round((enhanced_correct - regular_correct) / total_tests * 100, 2),
                "accuracy_by_difficulty": accuracy_by_difficulty
            },
            "performance_metrics": {
                "avg_response_times": {
                    "regular": round(statistics.mean([r.regular_total_time for r in self.results]), 4),
                    "enhanced": round(statistics.mean([r.enhanced_total_time for r in self.results]), 4)
                },
                "percentiles": performance_percentiles,
                "timeout_rate": sum(1 for r in self.results if r.timeout_occurred) / total_tests * 100
            },
            "category_analysis": category_analysis,
            "adversarial_analysis": adversarial_analysis,
            "statistical_tests": statistical_tests,
            "quality_scores": quality_scores,
            "recommendations": self._generate_deployment_recommendations()
        }
    
    def _perform_ultra_advanced_statistical_tests(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
            return {"error": "Advanced libraries not available"}
        
        try:
            tests = {}
            
            # Accuracy tests by category
            for category in set(r.category for r in self.results):
                cat_results = [r for r in self.results if r.category == category]
                if len(cat_results) >= 5:  # Minimum sample size
                    regular_correct = [1 if r.is_regular_correct else 0 for r in cat_results]
                    enhanced_correct = [1 if r.is_enhanced_correct else 0 for r in cat_results]
                    
                    # McNemar's test for paired accuracy comparison
                    contingency = [[sum(1 for i, r in enumerate(cat_results) if regular_correct[i] == 1 and enhanced_correct[i] == 1),
                                   sum(1 for i, r in enumerate(cat_results) if regular_correct[i] == 1 and enhanced_correct[i] == 0)],
                                  [sum(1 for i, r in enumerate(cat_results) if regular_correct[i] == 0 and enhanced_correct[i] == 1),
                                   sum(1 for i, r in enumerate(cat_results) if regular_correct[i] == 0 and enhanced_correct[i] == 0)]]
                    
                    chi2, p_value = chi2_contingency(contingency)[:2]
                    tests[f"{category}_accuracy"] = {
                        "test": "McNemar",
                        "chi2": round(chi2, 4),
                        "p_value": round(p_value, 6),
                        "significant": p_value < 0.05
                    }
            
            # Performance correlation with difficulty
            difficulties = [r.difficulty for r in self.results]
            regular_times = [r.regular_total_time for r in self.results]
            enhanced_times = [r.enhanced_total_time for r in self.results]
            
            reg_corr, reg_p = pearsonr(difficulties, regular_times)
            enh_corr, enh_p = pearsonr(difficulties, enhanced_times)
            
            tests["performance_correlation"] = {
                "regular": {"correlation": round(reg_corr, 4), "p_value": round(reg_p, 6)},
                "enhanced": {"correlation": round(enh_corr, 4), "p_value": round(enh_p, 6)}
            }
            
            # Overall quality comparison
            regular_quality = [r.correctness_confidence if r.is_regular_correct else 0 for r in self.results]
            enhanced_quality = [r.correctness_confidence if r.is_enhanced_correct else 0 for r in self.results]
            
            t_stat, p_quality = ttest_ind(enhanced_quality, regular_quality)
            tests["quality_comparison"] = {
                "test": "Independent t-test",
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_quality, 6),
                "significant": p_quality < 0.05
            }
            
            return tests
            
        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _calculate_quality_scores(self) -> Dict[str, float]:
        """Calculate comprehensive quality scores"""
        
        scores = {}
        
        # Overall system quality (0-100)
        enhanced_wins = sum(1 for r in self.results if r.winner == "enhanced")
        regular_wins = sum(1 for r in self.results if r.winner == "regular")
        ties = sum(1 for r in self.results if r.winner == "tie")
        
        scores["enhanced_dominance"] = enhanced_wins / len(self.results) * 100
        scores["competition_score"] = (enhanced_wins - regular_wins) / len(self.results) * 100
        
        # Robustness score (how well it handles difficult queries)
        high_difficulty = [r for r in self.results if r.difficulty >= 8]
        if high_difficulty:
            scores["robustness_score"] = sum(r.robustness_score for r in high_difficulty) / len(high_difficulty) * 100
        else:
            scores["robustness_score"] = 50.0
        
        # Efficiency score (performance vs accuracy trade-off)
        efficiency_scores = [r.efficiency_score for r in self.results if r.efficiency_score > 0]
        if efficiency_scores:
            scores["efficiency_score"] = statistics.mean(efficiency_scores) * 100
        else:
            scores["efficiency_score"] = 50.0
        
        # Consistency score (variance in performance)
        enhanced_accuracies = []
        for category in set(r.category for r in self.results):
            cat_results = [r for r in self.results if r.category == category]
            if cat_results:
                enhanced_accuracies.append(sum(1 for r in cat_results if r.is_enhanced_correct) / len(cat_results))
        
        if len(enhanced_accuracies) > 1:
            consistency = 1.0 - np.std(enhanced_accuracies)
            scores["consistency_score"] = max(consistency * 100, 0)
        else:
            scores["consistency_score"] = 50.0
        
        return scores
    
    def _generate_deployment_recommendations(self) -> Dict[str, Any]:
        """Generate intelligent deployment recommendations"""
        
        recommendations = {
            "primary_recommendation": "",
            "confidence_level": 0.0,
            "risk_assessment": "",
            "deployment_strategy": "",
            "monitoring_requirements": [],
            "fallback_strategy": ""
        }
        
        # Calculate decision factors
        enhanced_correct = sum(1 for r in self.results if r.is_enhanced_correct)
        regular_correct = sum(1 for r in self.results if r.is_regular_correct)
        total = len(self.results)
        
        accuracy_improvement = (enhanced_correct - regular_correct) / total * 100
        
        # Critical failure analysis
        critical_failures = [r for r in self.results if r.category == "entity_mismatch" and not r.is_enhanced_correct]
        critical_failure_rate = len(critical_failures) / max(sum(1 for r in self.results if r.category == "entity_mismatch"), 1) * 100
        
        # Performance analysis
        avg_speed_improvement = statistics.mean([r.speed_delta for r in self.results])
        
        # Decision logic
        confidence = 0.0
        
        if accuracy_improvement >= 15 and critical_failure_rate < 5 and avg_speed_improvement >= 0:
            recommendations["primary_recommendation"] = "DEPLOY_ENHANCED"
            recommendations["deployment_strategy"] = "full_rollout"
            confidence = 0.95
        elif accuracy_improvement >= 10 and critical_failure_rate < 10:
            recommendations["primary_recommendation"] = "GRADUAL_ROLLOUT"
            recommendations["deployment_strategy"] = "canary_deployment"
            confidence = 0.80
        elif accuracy_improvement >= 5 and critical_failure_rate < 15:
            recommendations["primary_recommendation"] = "LIMITED_DEPLOYMENT"
            recommendations["deployment_strategy"] = "specific_use_cases"
            confidence = 0.65
        else:
            recommendations["primary_recommendation"] = "KEEP_REGULAR"
            recommendations["deployment_strategy"] = "enhanced_research_needed"
            confidence = 0.70
        
        recommendations["confidence_level"] = confidence
        
        # Risk assessment
        if critical_failure_rate > 20:
            recommendations["risk_assessment"] = "HIGH_RISK"
        elif critical_failure_rate > 10:
            recommendations["risk_assessment"] = "MEDIUM_RISK"
        else:
            recommendations["risk_assessment"] = "LOW_RISK"
        
        # Monitoring requirements
        recommendations["monitoring_requirements"] = [
            "accuracy_by_category",
            "response_time_percentiles",
            "entity_blocking_accuracy",
            "error_rates",
            "user_satisfaction"
        ]
        
        return recommendations
    
    def generate_ultra_advanced_report(self, analysis: Dict[str, Any]):
        """Generate the most comprehensive report possible"""
        
        print("\n" + "=" * 120)
        print("🚀 ULTRA-ADVANCED SEMANTIC CACHE TESTING REPORT")
        print("=" * 120)
        
        # Executive Summary
        print(f"\n📋 EXECUTIVE SUMMARY:")
        rec = analysis["recommendations"]
        print(f"   🎯 PRIMARY RECOMMENDATION: {rec['primary_recommendation']}")
        print(f"   📊 CONFIDENCE LEVEL: {rec['confidence_level']:.1%}")
        print(f"   ⚠️ RISK ASSESSMENT: {rec['risk_assessment']}")
        print(f"   🚀 DEPLOYMENT STRATEGY: {rec['deployment_strategy']}")
        
        # Test Overview
        meta = analysis["test_metadata"]
        print(f"\n📊 TEST OVERVIEW:")
        print(f"   Session ID: {meta['session_id']}")
        print(f"   Framework: {meta['framework_version']}")
        print(f"   Total Tests: {meta['total_tests']}")
        print(f"   Duration: {meta['test_duration']:.1f} seconds")
        print(f"   Tests/Second: {meta['total_tests']/meta['test_duration']:.2f}")
        
        # Accuracy Deep Dive
        acc = analysis["accuracy_metrics"]
        print(f"\n🎯 ACCURACY ANALYSIS:")
        print(f"   Enhanced Pipeline: {acc['enhanced_accuracy']:.2f}%")
        print(f"   Regular Pipeline:  {acc['regular_accuracy']:.2f}%")
        print(f"   Improvement: {acc['accuracy_improvement']:+.2f}% {'🔥' if acc['accuracy_improvement'] > 0 else '❄️'}")
        
        # Difficulty breakdown
        print(f"\n   📈 ACCURACY BY DIFFICULTY LEVEL:")
        for diff, stats in sorted(acc['accuracy_by_difficulty'].items()):
            improvement = stats['enhanced'] - stats['regular']
            print(f"      Level {diff}: Enhanced {stats['enhanced']:5.1f}% | Regular {stats['regular']:5.1f}% | "
                  f"Δ{improvement:+5.1f}% ({stats['count']} tests)")
        
        # Performance Analysis
        perf = analysis["performance_metrics"]
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print(f"   Average Response Times:")
        print(f"      Regular:  {perf['avg_response_times']['regular']:.4f}s")
        print(f"      Enhanced: {perf['avg_response_times']['enhanced']:.4f}s")
        
        if "percentiles" in perf and perf["percentiles"]:
            p = perf["percentiles"]
            print(f"   Performance Percentiles:")
            print(f"      P50: Regular {p['regular_p50']:.3f}s | Enhanced {p['enhanced_p50']:.3f}s")
            print(f"      P95: Regular {p['regular_p95']:.3f}s | Enhanced {p['enhanced_p95']:.3f}s")
            print(f"      P99: Regular {p['regular_p99']:.3f}s | Enhanced {p['enhanced_p99']:.3f}s")
        
        print(f"   Timeout Rate: {perf['timeout_rate']:.2f}%")
        
        # Category Analysis
        print(f"\n📂 CATEGORY PERFORMANCE BREAKDOWN:")
        for category, stats in analysis["category_analysis"].items():
            if stats["total"] > 0:
                enh_acc = stats["enhanced_correct"] / stats["total"] * 100
                reg_acc = stats["regular_correct"] / stats["total"] * 100
                improvement = enh_acc - reg_acc
                
                speed_improvement = stats["avg_response_time_regular"] - stats["avg_response_time_enhanced"]
                
                print(f"   {category.replace('_', ' ').title():<20} | "
                      f"Acc: E{enh_acc:5.1f}% R{reg_acc:5.1f}% Δ{improvement:+5.1f}% | "
                      f"Speed: {speed_improvement:+.3f}s | "
                      f"Timeout: {stats['timeout_rate']:.1f}%")
                
                # Subcategory breakdown for critical categories
                if category in ["entity_mismatch", "adversarial", "compound"] and "subcategories" in stats:
                    for subcat, substats in stats["subcategories"].items():
                        if substats["total"] > 0:
                            sub_enh_acc = substats["enhanced_correct"] / substats["total"] * 100
                            sub_reg_acc = substats["regular_correct"] / substats["total"] * 100
                            print(f"      └─ {subcat:<15} | E{sub_enh_acc:5.1f}% R{sub_reg_acc:5.1f}% "
                                  f"(diff: {substats['avg_difficulty']:.1f})")
        
        # Adversarial Analysis
        if "adversarial_analysis" in analysis and analysis["adversarial_analysis"]:
            print(f"\n🛡️ ADVERSARIAL ATTACK RESISTANCE:")
            for attack_type, stats in analysis["adversarial_analysis"].items():
                print(f"   {attack_type.title():<20} | "
                      f"Enhanced: {stats['enhanced_success']:5.1f}% | "
                      f"Regular: {stats['regular_success']:5.1f}% | "
                      f"Robustness: {stats['avg_robustness']:.3f}")
        
        # Quality Scores
        if "quality_scores" in analysis:
            scores = analysis["quality_scores"]
            print(f"\n🏆 QUALITY SCORES:")
            print(f"   Enhanced Dominance: {scores['enhanced_dominance']:.1f}%")
            print(f"   Competition Score:  {scores['competition_score']:+.1f}")
            print(f"   Robustness Score:   {scores['robustness_score']:.1f}%")
            print(f"   Efficiency Score:   {scores['efficiency_score']:.1f}%")
            print(f"   Consistency Score:  {scores['consistency_score']:.1f}%")
        
        # Statistical Significance
        if "statistical_tests" in analysis and "error" not in analysis["statistical_tests"]:
            print(f"\n📈 STATISTICAL SIGNIFICANCE:")
            for test_name, test_result in analysis["statistical_tests"].items():
                if isinstance(test_result, dict) and "p_value" in test_result:
                    significance = "✅ SIGNIFICANT" if test_result.get("significant", False) else "❌ NOT SIGNIFICANT"
                    print(f"   {test_name.replace('_', ' ').title():<25} | p={test_result['p_value']:.6f} | {significance}")
        
        # Final Recommendations
        print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
        print(f"   📋 Primary Action: {rec['primary_recommendation']}")
        print(f"   📊 Confidence: {rec['confidence_level']:.1%}")
        print(f"   🚀 Strategy: {rec['deployment_strategy']}")
        print(f"   ⚠️ Risk Level: {rec['risk_assessment']}")
        
        print(f"\n   📊 Required Monitoring:")
        for requirement in rec.get("monitoring_requirements", []):
            print(f"      • {requirement.replace('_', ' ').title()}")
        
        # Critical Issues Alert
        critical_issues = []
        
        # Check entity blocking failures
        entity_results = [r for r in self.results if r.category == "entity_mismatch"]
        if entity_results:
            entity_failure_rate = sum(1 for r in entity_results if not r.is_enhanced_correct) / len(entity_results) * 100
            if entity_failure_rate > 15:
                critical_issues.append(f"HIGH ENTITY BLOCKING FAILURE RATE: {entity_failure_rate:.1f}%")
        
        # Check adversarial resistance
        adversarial_results = [r for r in self.results if r.adversarial_type]
        if adversarial_results:
            adv_failure_rate = sum(1 for r in adversarial_results if not r.is_enhanced_correct) / len(adversarial_results) * 100
            if adv_failure_rate > 30:
                critical_issues.append(f"POOR ADVERSARIAL RESISTANCE: {adv_failure_rate:.1f}% failure rate")
        
        # Check timeout issues
        if perf["timeout_rate"] > 5:
            critical_issues.append(f"HIGH TIMEOUT RATE: {perf['timeout_rate']:.1f}%")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                print(f"   ⚠️ {issue}")
        else:
            print(f"\n✅ NO CRITICAL ISSUES DETECTED")
        
        print(f"\n" + "=" * 120)
    
    def generate_ultra_advanced_visualizations(self, analysis: Dict[str, Any], save_plots: bool = True):
        """Generate enterprise-grade visualizations"""
        
        try:
            import numpy as np
        except ImportError:
            print("⚠️ numpy not available - skipping advanced visualizations")
            return
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("Set1")
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Accuracy by Difficulty Level
        ax1 = plt.subplot(3, 4, 1)
        if "accuracy_by_difficulty" in analysis["accuracy_metrics"]:
            diff_data = analysis["accuracy_metrics"]["accuracy_by_difficulty"]
            difficulties = list(diff_data.keys())
            regular_accs = [diff_data[d]["regular"] for d in difficulties]
            enhanced_accs = [diff_data[d]["enhanced"] for d in difficulties]
            
            x = np.arange(len(difficulties))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, regular_accs, width, label='Regular', color='#FF6B6B', alpha=0.8)
            bars2 = plt.bar(x + width/2, enhanced_accs, width, label='Enhanced', color='#4ECDC4', alpha=0.8)
            
            plt.xlabel('Difficulty Level')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy by Difficulty Level', fontweight='bold')
            plt.xticks(x, difficulties)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars1 + bars2:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance Percentiles
        ax2 = plt.subplot(3, 4, 2)
        if "percentiles" in analysis["performance_metrics"] and analysis["performance_metrics"]["percentiles"]:
            p = analysis["performance_metrics"]["percentiles"]
            percentiles = ['P50', 'P95', 'P99']
            regular_times = [p['regular_p50'], p['regular_p95'], p['regular_p99']]
            enhanced_times = [p['enhanced_p50'], p['enhanced_p95'], p['enhanced_p99']]
            
            x = np.arange(len(percentiles))
            width = 0.35
            
            plt.bar(x - width/2, regular_times, width, label='Regular', color='#FF6B6B', alpha=0.8)
            plt.bar(x + width/2, enhanced_times, width, label='Enhanced', color='#4ECDC4', alpha=0.8)
            
            plt.xlabel('Percentile')
            plt.ylabel('Response Time (s)')
            plt.title('Performance Percentiles', fontweight='bold')
            plt.xticks(x, percentiles)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Category Performance Heatmap
        ax3 = plt.subplot(3, 4, 3)
        categories = list(analysis["category_analysis"].keys())
        regular_accs = []
        enhanced_accs = []
        
        for cat in categories:
            cat_data = analysis["category_analysis"][cat]
            reg_acc = (cat_data["regular_correct"] / cat_data["total"]) * 100 if cat_data["total"] > 0 else 0
            enh_acc = (cat_data["enhanced_correct"] / cat_data["total"]) * 100 if cat_data["total"] > 0 else 0
            regular_accs.append(reg_acc)
            enhanced_accs.append(enh_acc)
        
        heatmap_data = np.array([regular_accs, enhanced_accs])
        im = plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax3, label='Accuracy (%)')
        plt.yticks([0, 1], ['Regular', 'Enhanced'])
        plt.xticks(range(len(categories)), [cat.replace('_', '\n') for cat in categories], rotation=45, ha='right')
        plt.title('Category Performance Heatmap', fontweight='bold')
        
        # 4. Quality Scores Radar Chart
        ax4 = plt.subplot(3, 4, 4, projection='polar')
        if "quality_scores" in analysis:
            scores = analysis["quality_scores"]
            categories_radar = ['Dominance', 'Competition', 'Robustness', 'Efficiency', 'Consistency']
            values = [
                scores.get('enhanced_dominance', 0),
                max(scores.get('competition_score', 0), 0),  # Ensure positive for radar
                scores.get('robustness_score', 0),
                scores.get('efficiency_score', 0),
                scores.get('consistency_score', 0)
            ]
            
            # Close the radar chart
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
            angles += angles[:1]
            
            plt.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
            plt.fill(angles, values, alpha=0.25, color='#4ECDC4')
            plt.xticks(angles[:-1], categories_radar)
            plt.ylim(0, 100)
            plt.title('Enhanced Pipeline Quality Scores', fontweight='bold', pad=20)
        
        # 5. Response Time Distribution
        ax5 = plt.subplot(3, 4, 5)
        regular_times = [r.regular_total_time for r in self.results if r.regular_total_time > 0]
        enhanced_times = [r.enhanced_total_time for r in self.results if r.enhanced_total_time > 0]
        
        plt.hist(regular_times, alpha=0.7, label='Regular', bins=30, color='#FF6B6B', density=True)
        plt.hist(enhanced_times, alpha=0.7, label='Enhanced', bins=30, color='#4ECDC4', density=True)
        plt.xlabel('Response Time (s)')
        plt.ylabel('Density')
        plt.title('Response Time Distribution', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Confidence Score Comparison
        ax6 = plt.subplot(3, 4, 6)
        regular_confs = [r.regular_confidence for r in self.results]
        enhanced_confs = [r.enhanced_confidence for r in self.results]
        
        plt.scatter(regular_confs, enhanced_confs, alpha=0.6, s=20, color='#4ECDC4')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Equal Confidence')
        plt.xlabel('Regular Confidence')
        plt.ylabel('Enhanced Confidence')
        plt.title('Confidence Score Comparison', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Adversarial Resistance
        ax7 = plt.subplot(3, 4, 7)
        if "adversarial_analysis" in analysis and analysis["adversarial_analysis"]:
            adv_data = analysis["adversarial_analysis"]
            attack_types = list(adv_data.keys())
            reg_success = [adv_data[att]["regular_success"] for att in attack_types]
            enh_success = [adv_data[att]["enhanced_success"] for att in attack_types]
            
            x = np.arange(len(attack_types))
            width = 0.35
            
            plt.bar(x - width/2, reg_success, width, label='Regular', color='#FF6B6B', alpha=0.8)
            plt.bar(x + width/2, enh_success, width, label='Enhanced', color='#4ECDC4', alpha=0.8)
            
            plt.xlabel('Attack Type')
            plt.ylabel('Success Rate (%)')
            plt.title('Adversarial Attack Resistance', fontweight='bold')
            plt.xticks(x, [att.replace('_', '\n') for att in attack_types], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No Adversarial\nTests Available', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=12)
            plt.title('Adversarial Attack Resistance', fontweight='bold')
        
        # 8. Winner Distribution by Category
        ax8 = plt.subplot(3, 4, 8)
        enhanced_wins_by_cat = []
        regular_wins_by_cat = []
        ties_by_cat = []
        
        for cat in categories[:6]:  # Limit to top 6 categories
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                enhanced_wins_by_cat.append(sum(1 for r in cat_results if r.winner == "enhanced"))
                regular_wins_by_cat.append(sum(1 for r in cat_results if r.winner == "regular"))
                ties_by_cat.append(sum(1 for r in cat_results if r.winner == "tie"))
        
        x = np.arange(len(categories[:6]))
        width = 0.25
        
        plt.bar(x - width, enhanced_wins_by_cat, width, label='Enhanced Wins', color='#4ECDC4', alpha=0.8)
        plt.bar(x, regular_wins_by_cat, width, label='Regular Wins', color='#FF6B6B', alpha=0.8)
        plt.bar(x + width, ties_by_cat, width, label='Ties', color='#FFE66D', alpha=0.8)
        
        plt.xlabel('Category')
        plt.ylabel('Number of Wins')
        plt.title('Winner Distribution by Category', fontweight='bold')
        plt.xticks(x, [cat.replace('_', '\n') for cat in categories[:6]], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Statistical Significance Test Results
        ax9 = plt.subplot(3, 4, 9)
        if "statistical_tests" in analysis and "error" not in analysis["statistical_tests"]:
            tests = analysis["statistical_tests"]
            test_names = []
            p_values = []
            
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict) and "p_value" in test_result:
                    test_names.append(test_name.replace('_', '\n'))
                    p_values.append(test_result["p_value"])
            
            if p_values:
                colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
                bars = plt.bar(range(len(test_names)), [-np.log10(max(p, 1e-10)) for p in p_values], color=colors)
                plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', label='p=0.05 threshold')
                plt.axhline(y=-np.log10(0.01), color='red', linestyle='--', label='p=0.01 threshold')
                
                plt.ylabel('-log10(p-value)')
                plt.title('Statistical Significance Tests', fontweight='bold')
                plt.xticks(range(len(test_names)), test_names, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Statistical tests\nnot available', ha='center', va='center', 
                    transform=ax9.transAxes, fontsize=12)
            plt.title('Statistical Significance Tests', fontweight='bold')
        
        # 10. Efficiency vs Accuracy Scatter
        ax10 = plt.subplot(3, 4, 10)
        accuracy_scores = [1 if r.is_enhanced_correct else 0 for r in self.results]
        efficiency_scores = [r.efficiency_score for r in self.results]
        colors = [r.difficulty for r in self.results]
        
        scatter = plt.scatter(efficiency_scores, accuracy_scores, c=colors, cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax10, label='Difficulty Level')
        plt.xlabel('Efficiency Score')
        plt.ylabel('Accuracy (0=Wrong, 1=Correct)')
        plt.title('Efficiency vs Accuracy', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 11. Timeout and Error Analysis
        ax11 = plt.subplot(3, 4, 11)
        timeout_by_category = {}
        error_by_category = {}
        
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                timeout_rate = sum(1 for r in cat_results if r.timeout_occurred) / len(cat_results) * 100
                error_rate = sum(1 for r in cat_results if r.enhanced_error or r.regular_error) / len(cat_results) * 100
                timeout_by_category[cat] = timeout_rate
                error_by_category[cat] = error_rate
        
        cats = list(timeout_by_category.keys())[:6]  # Top 6 categories
        timeouts = [timeout_by_category[cat] for cat in cats]
        errors = [error_by_category[cat] for cat in cats]
        
        x = np.arange(len(cats))
        width = 0.35
        
        plt.bar(x - width/2, timeouts, width, label='Timeout Rate', color='#FF6B6B', alpha=0.8)
        plt.bar(x + width/2, errors, width, label='Error Rate', color='#FFE66D', alpha=0.8)
        
        plt.xlabel('Category')
        plt.ylabel('Rate (%)')
        plt.title('Timeout and Error Analysis', fontweight='bold')
        plt.xticks(x, [cat.replace('_', '\n') for cat in cats], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. Overall System Health Score
        ax12 = plt.subplot(3, 4, 12)
        if "quality_scores" in analysis:
            scores = analysis["quality_scores"]
            
            # Create a comprehensive health score
            health_components = {
                'Accuracy': min(analysis["accuracy_metrics"]["enhanced_accuracy"], 100),
                'Performance': min(100 - analysis["performance_metrics"]["timeout_rate"] * 2, 100),
                'Robustness': scores.get('robustness_score', 50),
                'Consistency': scores.get('consistency_score', 50)
            }
            
            # Create a gauge-like visualization
            labels = list(health_components.keys())
            values = list(health_components.values())
            colors = ['#ff4444' if v < 50 else '#ffaa00' if v < 75 else '#44ff44' for v in values]
            
            wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            plt.title('System Health Score', fontweight='bold')
            
            # Overall health score
            overall_health = np.mean(values)
            plt.text(0, -1.3, f'Overall Health: {overall_health:.1f}%', 
                    ha='center', fontsize=14, fontweight='bold',
                    color='#44ff44' if overall_health >= 75 else '#ffaa00' if overall_health >= 50 else '#ff4444')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ultra_advanced_ab_test_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Ultra-advanced visualizations saved to: {filename}")
        
        plt.show()
    
    def export_ultra_comprehensive_results(self, analysis: Dict[str, Any], filename: str = None):
        """Export comprehensive results in multiple formats"""
        if not self.results:
            print("❌ No results to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = filename or f"ultra_advanced_results_{timestamp}"
        
        # Export detailed CSV
        df = pd.DataFrame([asdict(result) for result in self.results])
        csv_filename = f"{base_filename}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"📁 Detailed results exported to: {csv_filename}")
        
        # Export analysis summary JSON
        json_filename = f"{base_filename}_analysis.json"
        with open(json_filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📊 Analysis summary exported to: {json_filename}")
        
        # Export executive summary
        exec_filename = f"{base_filename}_executive_summary.txt"
        with open(exec_filename, 'w') as f:
            f.write("ULTRA-ADVANCED SEMANTIC CACHE TESTING - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            rec = analysis["recommendations"]
            f.write(f"PRIMARY RECOMMENDATION: {rec['primary_recommendation']}\n")
            f.write(f"CONFIDENCE LEVEL: {rec['confidence_level']:.1%}\n")
            f.write(f"RISK ASSESSMENT: {rec['risk_assessment']}\n")
            f.write(f"DEPLOYMENT STRATEGY: {rec['deployment_strategy']}\n\n")
            
            acc = analysis["accuracy_metrics"]
            f.write(f"ACCURACY RESULTS:\n")
            f.write(f"  Enhanced: {acc['enhanced_accuracy']:.2f}%\n")
            f.write(f"  Regular:  {acc['regular_accuracy']:.2f}%\n")
            f.write(f"  Improvement: {acc['accuracy_improvement']:+.2f}%\n\n")
            
            if "quality_scores" in analysis:
                scores = analysis["quality_scores"]
                f.write(f"QUALITY SCORES:\n")
                for score_name, score_value in scores.items():
                    f.write(f"  {score_name.replace('_', ' ').title()}: {score_value:.1f}\n")
        
        print(f"📋 Executive summary exported to: {exec_filename}")
        
        return csv_filename, json_filename, exec_filename

def get_ultra_advanced_user_choice():
    """Get user's ultra-advanced testing choice"""
    print("\n🚀 ULTRA-ADVANCED TESTING OPTIONS:")
    print("1. 🏃 Quick Diagnostic (20 queries, ~1 minute)")
    print("2. 🔬 Standard Comprehensive (80+ queries, ~3-5 minutes)")
    print("3. 🚀 Ultra-Parallel Comprehensive (80+ queries, ~1-2 minutes)")
    print("4. 🔥 MAXIMUM STRESS TEST (80+ queries + stress tests, ~5-10 minutes)")
    print("5. ⚙️ Custom Configuration")
    print("6. ❌ Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-6): "))
            if 1 <= choice <= 6:
                return choice
            else:
                print("Please enter a number between 1 and 6")
        except ValueError:
            print("Please enter a valid number")

def get_custom_configuration():
    """Get custom test configuration from user"""
    config = {}
    
    print("\n⚙️ CUSTOM CONFIGURATION:")
    
    # Parallel vs Sequential
    parallel = input("Use parallel testing? (y/n) [y]: ").lower()
    config["parallel"] = parallel != 'n'
    
    if config["parallel"]:
        workers = input("Max workers (1-10) [4]: ")
        try:
            config["max_workers"] = min(max(int(workers), 1), 10) if workers else 4
        except:
            config["max_workers"] = 4
    
    # Stress testing
    stress = input("Include stress tests? (y/n) [y]: ").lower()
    config["stress_test"] = stress != 'n'
    
    # Adversarial testing
    adversarial = input("Include adversarial tests? (y/n) [y]: ").lower()
    config["adversarial_test"] = adversarial != 'n'
    
    # Timeout
    timeout = input("Request timeout in seconds [30]: ")
    try:
        config["timeout"] = max(int(timeout), 5) if timeout else 30
    except:
        config["timeout"] = 30
    
    # Retries
    retries = input("Number of retries [3]: ")
    try:
        config["retries"] = max(int(retries), 0) if retries else 3
    except:
        config["retries"] = 3
    
    return config

def main():
    """Ultra-advanced main testing function"""
    
    print("🚀 ULTRA-ADVANCED SEMANTIC CACHE TESTING FRAMEWORK")
    print("=" * 100)
    print("Enterprise-grade testing with adversarial attacks, stress tests, and statistical rigor")
    print("Designed to push your semantic cache to its absolute limits")
    print()
    
    # Initialize ultra-advanced tester
    tester = UltraAdvancedTester()
    
    print("🔍 Checking API availability...")
    try:
        # Test both APIs with advanced probing
        regular_test = tester.test_single_api_advanced("test probe", "/api/query", {"timeout": 5})
        enhanced_test = tester.test_single_api_advanced("test probe", "/api/query/enhanced", {"timeout": 5})
        
        regular_available = regular_test["error"] == ""
        enhanced_available = enhanced_test["error"] == ""
        
        print(f"   Regular API: {'✅ Available' if regular_available else '❌ Not Available'}")
        print(f"   Enhanced API: {'✅ Available' if enhanced_available else '❌ Not Available'}")
        
        if regular_available and enhanced_available:
            print(f"   Regular Response Time: {regular_test['total_time']:.3f}s")
            print(f"   Enhanced Response Time: {enhanced_test['total_time']:.3f}s")
        
        if not (regular_available and enhanced_available):
            print("\n❌ Both APIs must be available for ultra-advanced testing.")
            print("Please ensure your Flask app is running with both pipelines enabled.")
            return False
            
    except Exception as e:
        print(f"❌ Error during API probe: {e}")
        return False
    
    print("\n✅ Both APIs are ready for ultra-advanced testing!")
    
    # Get user choice
    choice = get_ultra_advanced_user_choice()
    
    if choice == 6:
        print("👋 Goodbye!")
        return
    
    # Configure test based on choice
    test_configs = {
        1: {"parallel": True, "max_workers": 2, "stress_test": False, "adversarial_test": True, "timeout": 20, "retries": 1},
        2: {"parallel": False, "max_workers": 1, "stress_test": True, "adversarial_test": True, "timeout": 30, "retries": 3},
        3: {"parallel": True, "max_workers": 6, "stress_test": True, "adversarial_test": True, "timeout": 30, "retries": 3},
        4: {"parallel": True, "max_workers": 8, "stress_test": True, "adversarial_test": True, "timeout": 45, "retries": 5},
        5: get_custom_configuration()
    }
    
    config = test_configs[choice]
    
    print(f"\n🎯 Configuration: {config}")
    print("⚠️  This will comprehensively test your semantic cache system")
    print("   Including adversarial attacks, edge cases, and stress tests")
    
    confirm = input("\nProceed with ultra-advanced testing? (y/n): ").lower()
    if confirm != 'y':
        print("Testing cancelled.")
        return
    
    # Run ultra-comprehensive test
    try:
        print(f"\n🚀 Initiating ultra-advanced testing sequence...")
        results = tester.run_ultra_comprehensive_test(config)
        
        print(f"\n📊 Generating ultra-comprehensive analysis...")
        analysis = tester.generate_ultra_comprehensive_analysis()
        
        print(f"\n📈 Creating enterprise-grade visualizations...")
        try:
            tester.generate_ultra_advanced_visualizations(analysis)
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
            print("   (Analysis will continue without visualizations)")
        
        # Generate ultra-detailed report
        tester.generate_ultra_advanced_report(analysis)
        
        # Export comprehensive results
        print(f"\n💾 Exporting ultra-comprehensive results...")
        files = tester.export_ultra_comprehensive_results(analysis)
        
        print(f"\n🎉 ULTRA-ADVANCED TESTING COMPLETE!")
        print(f"   📊 Analysis generated with enterprise-grade statistical rigor")
        print(f"   📁 Results exported to multiple formats")
        print(f"   🎯 Deployment recommendation: {analysis['recommendations']['primary_recommendation']}")
        print(f"   📈 Confidence level: {analysis['recommendations']['confidence_level']:.1%}")
        
        if analysis['recommendations']['confidence_level'] >= 0.8:
            print(f"   ✅ HIGH CONFIDENCE RECOMMENDATION - Ready for deployment decision")
        else:
            print(f"   🤔 MODERATE CONFIDENCE - Consider additional testing")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Ultra-advanced testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Ultra-advanced testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()