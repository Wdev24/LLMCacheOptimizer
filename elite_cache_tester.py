#!/usr/bin/env python3
"""
Elite Advanced Semantic Cache Testing System
Tests for false positives, edge cases, and system efficiency with 100+ prompts
"""

import requests
import json
import time
import statistics
from typing import Dict, List, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime

class EliteCacheTester:
    """Advanced testing system for semantic cache validation"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        self.false_positives = []
        self.false_negatives = []
        self.performance_metrics = {}
        
    def get_test_prompts(self) -> List[Dict[str, any]]:
        """Generate 100+ test prompts with expected behaviors"""
        
        test_prompts = [
            # === EXACT MATCHES (Should be CACHE HITS) ===
            {"prompt": "What is machine learning?", "expected": "cache", "category": "exact_match", "difficulty": 1},
            {"prompt": "Define artificial intelligence", "expected": "cache", "category": "exact_match", "difficulty": 1},
            {"prompt": "How does neural network work?", "expected": "cache", "category": "exact_match", "difficulty": 1},
            {"prompt": "What is LLM?", "expected": "cache", "category": "exact_match", "difficulty": 1},
            
            # === SEMANTIC EQUIVALENTS (Should be CACHE HITS) ===
            {"prompt": "What is ML?", "expected": "cache", "category": "semantic_match", "difficulty": 2},
            {"prompt": "Explain machine learning", "expected": "cache", "category": "semantic_match", "difficulty": 2},
            {"prompt": "Tell me about artificial intelligence", "expected": "cache", "category": "semantic_match", "difficulty": 2},
            {"prompt": "How do neural networks function?", "expected": "cache", "category": "semantic_match", "difficulty": 2},
            {"prompt": "Define LLM", "expected": "cache", "category": "semantic_match", "difficulty": 2},
            {"prompt": "What are large language models?", "expected": "cache", "category": "semantic_match", "difficulty": 2},
            
            # === CRITICAL FALSE POSITIVE TESTS ===
            # Similar but DIFFERENT entities/numbers
            {"prompt": "Call Ramesh", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Ramesh Gupta", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Ramesh Sharma", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Ramesh Patel", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Kumar", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Singh", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            
            # Time/number precision tests
            {"prompt": "Set alarm for 10:49", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 10:51", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 10:52", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 10:53", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 10:54", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 10:56", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 11:50", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 9:50", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            
            # === INTENT MISMATCH TESTS ===
            {"prompt": "Machine learning stocks to buy", "expected": "llm", "category": "intent_mismatch", "difficulty": 4},
            {"prompt": "Machine learning job salary", "expected": "llm", "category": "intent_mismatch", "difficulty": 4},
            {"prompt": "AI companies list", "expected": "llm", "category": "intent_mismatch", "difficulty": 4},
            {"prompt": "Neural network hardware requirements", "expected": "llm", "category": "intent_mismatch", "difficulty": 4},
            {"prompt": "LLM API pricing", "expected": "llm", "category": "intent_mismatch", "difficulty": 4},
            
            # === CONTEXT SWITCH TESTS ===
            {"prompt": "What is machine learning in finance?", "expected": "llm", "category": "context_switch", "difficulty": 4},
            {"prompt": "AI in healthcare applications", "expected": "llm", "category": "context_switch", "difficulty": 4},
            {"prompt": "Neural networks for image recognition", "expected": "llm", "category": "context_switch", "difficulty": 4},
            {"prompt": "LLM for code generation", "expected": "llm", "category": "context_switch", "difficulty": 4},
            
            # === NEGATION TESTS ===
            {"prompt": "What is NOT machine learning?", "expected": "llm", "category": "negation", "difficulty": 4},
            {"prompt": "Why AI is not intelligent?", "expected": "llm", "category": "negation", "difficulty": 4},
            {"prompt": "Neural networks disadvantages", "expected": "llm", "category": "negation", "difficulty": 4},
            
            # === QUESTION TYPE VARIATIONS ===
            {"prompt": "Is machine learning useful?", "expected": "llm", "category": "question_type", "difficulty": 3},
            {"prompt": "Can AI replace humans?", "expected": "llm", "category": "question_type", "difficulty": 3},
            {"prompt": "Should I learn neural networks?", "expected": "llm", "category": "question_type", "difficulty": 3},
            {"prompt": "Will LLMs get better?", "expected": "llm", "category": "question_type", "difficulty": 3},
            
            # === TYPOS AND VARIATIONS ===
            {"prompt": "What is machien learning?", "expected": "cache", "category": "typo_test", "difficulty": 3},
            {"prompt": "Defien artificial intelligence", "expected": "cache", "category": "typo_test", "difficulty": 3},
            {"prompt": "How does nueral network work?", "expected": "cache", "category": "typo_test", "difficulty": 3},
            
            # === LANGUAGE STYLE TESTS ===
            {"prompt": "Machine learning definition please", "expected": "cache", "category": "style_variation", "difficulty": 2},
            {"prompt": "Can you explain what AI is?", "expected": "cache", "category": "style_variation", "difficulty": 2},
            {"prompt": "I need to understand neural networks", "expected": "cache", "category": "style_variation", "difficulty": 2},
            {"prompt": "Help me understand LLMs", "expected": "cache", "category": "style_variation", "difficulty": 2},
            
            # === COMPOUND QUESTIONS ===
            {"prompt": "What is machine learning and how does it work?", "expected": "llm", "category": "compound", "difficulty": 4},
            {"prompt": "Define AI and give examples", "expected": "llm", "category": "compound", "difficulty": 4},
            {"prompt": "Neural networks vs deep learning differences", "expected": "llm", "category": "compound", "difficulty": 4},
            
            # === COMPARISON TESTS ===
            {"prompt": "Machine learning vs artificial intelligence", "expected": "llm", "category": "comparison", "difficulty": 4},
            {"prompt": "Deep learning vs neural networks", "expected": "llm", "category": "comparison", "difficulty": 4},
            {"prompt": "LLM vs transformer difference", "expected": "llm", "category": "comparison", "difficulty": 4},
            
            # === SPECIFIC DETAILS TESTS ===
            {"prompt": "Machine learning algorithms list", "expected": "llm", "category": "specific_details", "difficulty": 4},
            {"prompt": "AI programming languages", "expected": "llm", "category": "specific_details", "difficulty": 4},
            {"prompt": "Neural network layers explanation", "expected": "llm", "category": "specific_details", "difficulty": 4},
            {"prompt": "LLM training process", "expected": "llm", "category": "specific_details", "difficulty": 4},
            
            # === COMPLETELY DIFFERENT TOPICS ===
            {"prompt": "How to cook pasta?", "expected": "llm", "category": "different_topic", "difficulty": 1},
            {"prompt": "Weather forecast today", "expected": "llm", "category": "different_topic", "difficulty": 1},
            {"prompt": "Best restaurants nearby", "expected": "llm", "category": "different_topic", "difficulty": 1},
            {"prompt": "Stock market trends", "expected": "llm", "category": "different_topic", "difficulty": 1},
            {"prompt": "How to lose weight?", "expected": "llm", "category": "different_topic", "difficulty": 1},
            
            # === EDGE CASE TESTS ===
            {"prompt": "", "expected": "error", "category": "edge_case", "difficulty": 5},
            {"prompt": "   ", "expected": "error", "category": "edge_case", "difficulty": 5},
            {"prompt": "a", "expected": "llm", "category": "edge_case", "difficulty": 5},
            {"prompt": "?" * 100, "expected": "llm", "category": "edge_case", "difficulty": 5},
            {"prompt": "machine learning " * 50, "expected": "llm", "category": "edge_case", "difficulty": 5},
            
            # === AMBIGUOUS QUERIES ===
            {"prompt": "learning", "expected": "llm", "category": "ambiguous", "difficulty": 4},
            {"prompt": "intelligence", "expected": "llm", "category": "ambiguous", "difficulty": 4},
            {"prompt": "network", "expected": "llm", "category": "ambiguous", "difficulty": 4},
            {"prompt": "model", "expected": "llm", "category": "ambiguous", "difficulty": 4},
            
            # === MULTILINGUAL TESTS ===
            {"prompt": "¿Qué es machine learning?", "expected": "llm", "category": "multilingual", "difficulty": 3},
            {"prompt": "Was ist künstliche Intelligenz?", "expected": "llm", "category": "multilingual", "difficulty": 3},
            {"prompt": "Qu'est-ce que l'IA?", "expected": "llm", "category": "multilingual", "difficulty": 3},
            
            # === CONVERSATIONAL CONTEXT ===
            {"prompt": "Thanks for explaining ML", "expected": "llm", "category": "conversational", "difficulty": 3},
            {"prompt": "That's helpful about AI", "expected": "llm", "category": "conversational", "difficulty": 3},
            {"prompt": "Can you tell me more?", "expected": "llm", "category": "conversational", "difficulty": 3},
            
            # === INSTRUCTION vs QUESTION TESTS ===
            {"prompt": "Teach me machine learning", "expected": "llm", "category": "instruction", "difficulty": 3},
            {"prompt": "Show me AI examples", "expected": "llm", "category": "instruction", "difficulty": 3},
            {"prompt": "List neural network types", "expected": "llm", "category": "instruction", "difficulty": 3},
            
            # === TEMPORAL CONTEXT ===
            {"prompt": "Latest machine learning trends", "expected": "llm", "category": "temporal", "difficulty": 4},
            {"prompt": "Future of AI", "expected": "llm", "category": "temporal", "difficulty": 4},
            {"prompt": "Recent neural network breakthroughs", "expected": "llm", "category": "temporal", "difficulty": 4},
            
            # === ADDITIONAL ENTITY TESTS ===
            {"prompt": "Call John Kumar", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Ramesh Martin", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Call Kumar Singh", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set alarm for 10:59", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set timer for 10:50", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            {"prompt": "Set reminder for 10:55", "expected": "llm", "category": "false_positive_test", "difficulty": 5},
            
            # === SEMANTIC BOUNDARY TESTS ===
            {"prompt": "Machine learning applications", "expected": "llm", "category": "semantic_boundary", "difficulty": 4},
            {"prompt": "AI ethics concerns", "expected": "llm", "category": "semantic_boundary", "difficulty": 4},
            {"prompt": "Neural network optimization", "expected": "llm", "category": "semantic_boundary", "difficulty": 4},
            {"prompt": "LLM limitations", "expected": "llm", "category": "semantic_boundary", "difficulty": 4},
            
            # === STRESS TESTS ===
            {"prompt": "What is machine learning? Please explain in detail with examples and applications.", "expected": "llm", "category": "stress_test", "difficulty": 4},
            {"prompt": "AI definition comprehensive guide", "expected": "llm", "category": "stress_test", "difficulty": 4},
        ]
        
        return test_prompts
    
    def send_query(self, prompt: str) -> Dict[str, any]:
        """Send a single query to the cache system"""
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/query",
                json={"query": prompt},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                data["response_time"] = end_time - start_time
                return data
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time": end_time - start_time,
                    "source": "error"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "response_time": 0,
                "source": "error"
            }
    
    def evaluate_result(self, test_case: Dict, result: Dict) -> Dict[str, any]:
        """Evaluate if the result matches expectations"""
        prompt = test_case["prompt"]
        expected = test_case["expected"]
        actual_source = result.get("source", "error")
        confidence = result.get("confidence", 0)
        
        # Determine if result is correct
        if expected == "error":
            is_correct = actual_source == "error"
        elif expected == "cache":
            is_correct = actual_source == "cache" and confidence >= 0.6
        elif expected == "llm":
            is_correct = actual_source == "llm" or confidence < 0.6
        else:
            is_correct = False
        
        # Classify error types
        error_type = None
        if not is_correct:
            if expected == "llm" and actual_source == "cache":
                error_type = "false_positive"  # Cache hit when should be LLM
                self.false_positives.append({
                    "prompt": prompt,
                    "expected": expected,
                    "actual": actual_source,
                    "confidence": confidence,
                    "category": test_case["category"]
                })
            elif expected == "cache" and actual_source == "llm":
                error_type = "false_negative"  # LLM when should be cache
                self.false_negatives.append({
                    "prompt": prompt,
                    "expected": expected,
                    "actual": actual_source,
                    "confidence": confidence,
                    "category": test_case["category"]
                })
            else:
                error_type = "other_error"
        
        return {
            "prompt": prompt,
            "expected": expected,
            "actual_source": actual_source,
            "confidence": confidence,
            "is_correct": is_correct,
            "error_type": error_type,
            "category": test_case["category"],
            "difficulty": test_case["difficulty"],
            "response_time": result.get("response_time", 0),
            "processing_time": result.get("processing_time", 0),
            "similarity_score": result.get("similarity_score", 0),
            "reranker_score": result.get("reranker_score", 0),
            "intent_score": result.get("intent_score", 0)
        }
    
    def run_single_test(self, test_case: Dict) -> Dict[str, any]:
        """Run a single test case"""
        print(f"Testing: '{test_case['prompt'][:50]}...' (Category: {test_case['category']})")
        
        result = self.send_query(test_case["prompt"])
        evaluation = self.evaluate_result(test_case, result)
        
        status = "✅" if evaluation["is_correct"] else "❌"
        print(f"  {status} Expected: {test_case['expected']}, Got: {evaluation['actual_source']} (Confidence: {evaluation['confidence']:.3f})")
        
        return evaluation
    
    def run_sequential_test(self) -> List[Dict[str, any]]:
        """Run tests sequentially"""
        test_prompts = self.get_test_prompts()
        results = []
        
        print(f"🚀 Running {len(test_prompts)} sequential tests...")
        print("=" * 80)
        
        for i, test_case in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}]", end=" ")
            result = self.run_single_test(test_case)
            results.append(result)
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.1)
        
        return results
    
    def run_parallel_test(self, max_workers: int = 5) -> List[Dict[str, any]]:
        """Run tests in parallel"""
        test_prompts = self.get_test_prompts()
        results = []
        
        print(f"🚀 Running {len(test_prompts)} parallel tests (workers: {max_workers})...")
        print("=" * 80)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.run_single_test, test_case) for test_case in test_prompts]
            
            for i, future in enumerate(futures, 1):
                print(f"\r[{i}/{len(test_prompts)}] Processing...", end="", flush=True)
                result = future.result()
                results.append(result)
        
        print("\n")
        return results
    
    def analyze_results(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Analyze test results and generate comprehensive metrics"""
        
        total_tests = len(results)
        correct_tests = sum(1 for r in results if r["is_correct"])
        
        # Basic metrics
        accuracy = (correct_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Error analysis
        false_positive_count = len(self.false_positives)
        false_negative_count = len(self.false_negatives)
        
        # Performance metrics
        response_times = [r["response_time"] for r in results if r["response_time"] > 0]
        processing_times = [r["processing_time"] for r in results if r["processing_time"] > 0]
        
        # Category analysis
        category_stats = {}
        for result in results:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0}
            category_stats[category]["total"] += 1
            if result["is_correct"]:
                category_stats[category]["correct"] += 1
        
        # Difficulty analysis
        difficulty_stats = {}
        for result in results:
            difficulty = result["difficulty"]
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"total": 0, "correct": 0}
            difficulty_stats[difficulty]["total"] += 1
            if result["is_correct"]:
                difficulty_stats[difficulty]["correct"] += 1
        
        # Confidence analysis
        confidence_scores = [r["confidence"] for r in results if r["confidence"] >= 0]
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "correct_predictions": correct_tests,
            "accuracy_percentage": round(accuracy, 2),
            "false_positives": false_positive_count,
            "false_negatives": false_negative_count,
            "false_positive_rate": round((false_positive_count / total_tests) * 100, 2),
            "false_negative_rate": round((false_negative_count / total_tests) * 100, 2),
            "performance": {
                "avg_response_time": round(statistics.mean(response_times), 3) if response_times else 0,
                "median_response_time": round(statistics.median(response_times), 3) if response_times else 0,
                "max_response_time": round(max(response_times), 3) if response_times else 0,
                "min_response_time": round(min(response_times), 3) if response_times else 0,
                "avg_processing_time": round(statistics.mean(processing_times), 3) if processing_times else 0,
            },
            "confidence_analysis": {
                "avg_confidence": round(statistics.mean(confidence_scores), 3) if confidence_scores else 0,
                "median_confidence": round(statistics.median(confidence_scores), 3) if confidence_scores else 0,
                "confidence_std": round(statistics.stdev(confidence_scores), 3) if len(confidence_scores) > 1 else 0,
            },
            "category_performance": {
                cat: {
                    "accuracy": round((stats["correct"] / stats["total"]) * 100, 2),
                    "total": stats["total"],
                    "correct": stats["correct"]
                }
                for cat, stats in category_stats.items()
            },
            "difficulty_performance": {
                diff: {
                    "accuracy": round((stats["correct"] / stats["total"]) * 100, 2),
                    "total": stats["total"],
                    "correct": stats["correct"]
                }
                for diff, stats in difficulty_stats.items()
            }
        }
        
        return analysis
    
    def generate_report(self, results: List[Dict[str, any]], analysis: Dict[str, any]):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 80)
        print("🏆 ELITE SEMANTIC CACHE TESTING REPORT")
        print("=" * 80)
        
        # Overall Performance
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {analysis['total_tests']}")
        print(f"   Accuracy: {analysis['accuracy_percentage']}%")
        print(f"   False Positives: {analysis['false_positives']} ({analysis['false_positive_rate']}%)")
        print(f"   False Negatives: {analysis['false_negatives']} ({analysis['false_negative_rate']}%)")
        
        # Performance Metrics
        perf = analysis['performance']
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Avg Response Time: {perf['avg_response_time']}s")
        print(f"   Median Response Time: {perf['median_response_time']}s")
        print(f"   Response Time Range: {perf['min_response_time']}s - {perf['max_response_time']}s")
        print(f"   Avg Processing Time: {perf['avg_processing_time']}s")
        
        # Confidence Analysis
        conf = analysis['confidence_analysis']
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   Average Confidence: {conf['avg_confidence']}")
        print(f"   Median Confidence: {conf['median_confidence']}")
        print(f"   Confidence Std Dev: {conf['confidence_std']}")
        
        # Category Performance
        print(f"\n📂 CATEGORY PERFORMANCE:")
        for category, stats in analysis['category_performance'].items():
            print(f"   {category.replace('_', ' ').title()}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
        
        # Difficulty Analysis
        print(f"\n🎚️ DIFFICULTY ANALYSIS:")
        for difficulty, stats in analysis['difficulty_performance'].items():
            print(f"   Level {difficulty}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
        
        # Critical False Positives
        if self.false_positives:
            print(f"\n🚨 CRITICAL FALSE POSITIVES (Cache when should be LLM):")
            for fp in self.false_positives[:10]:  # Show first 10
                print(f"   '{fp['prompt'][:60]}...' (Confidence: {fp['confidence']:.3f})")
        
        # False Negatives
        if self.false_negatives:
            print(f"\n⚠️ FALSE NEGATIVES (LLM when should be Cache):")
            for fn in self.false_negatives[:5]:  # Show first 5
                print(f"   '{fn['prompt'][:60]}...' (Confidence: {fn['confidence']:.3f})")
        
        # System Grade
        grade = self._calculate_grade(analysis)
        print(f"\n🏅 SYSTEM GRADE: {grade}")
        
        # Recommendations
        self._generate_recommendations(analysis)
    
    def _calculate_grade(self, analysis: Dict[str, any]) -> str:
        """Calculate system grade based on performance"""
        accuracy = analysis['accuracy_percentage']
        fp_rate = analysis['false_positive_rate']
        
        # Weighted scoring (accuracy 70%, false positive rate 30%)
        score = (accuracy * 0.7) - (fp_rate * 0.3)
        
        if score >= 95:
            return "A+ (Elite)"
        elif score >= 90:
            return "A (Excellent)"
        elif score >= 85:
            return "B+ (Very Good)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 75:
            return "C+ (Fair)"
        elif score >= 70:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
    
    def _generate_recommendations(self, analysis: Dict[str, any]):
        """Generate system improvement recommendations"""
        print(f"\n💡 RECOMMENDATIONS:")
        
        fp_rate = analysis['false_positive_rate']
        fn_rate = analysis['false_negative_rate']
        accuracy = analysis['accuracy_percentage']
        
        if fp_rate > 5:
            print(f"   🔧 High false positive rate ({fp_rate}%) - Consider:")
            print(f"      - Increasing confidence thresholds")
            print(f"      - Improving entity extraction")
            print(f"      - Enhanced intent classification")
        
        if fn_rate > 10:
            print(f"   🔧 High false negative rate ({fn_rate}%) - Consider:")
            print(f"      - Lowering confidence thresholds")
            print(f"      - Better semantic matching")
            print(f"      - More training data")
        
        if accuracy < 85:
            print(f"   🔧 Overall accuracy needs improvement ({accuracy}%) - Consider:")
            print(f"      - Model fine-tuning")
            print(f"      - Better preprocessing")
            print(f"      - Ensemble methods")
        
        avg_response_time = analysis['performance']['avg_response_time']
        if avg_response_time > 1.0:
            print(f"   ⚡ Response time optimization needed ({avg_response_time}s) - Consider:")
            print(f"      - Index optimization")
            print(f"      - Caching strategies")
            print(f"      - Parallel processing")
    
    def save_results(self, results: List[Dict[str, any]], analysis: Dict[str, any]):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f"test_results_{timestamp}.json", 'w') as f:
            json.dump({
                "analysis": analysis,
                "detailed_results": results,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives
            }, f, indent=2)
        
        # Save CSV for analysis
        df = pd.DataFrame(results)
        df.to_csv(f"test_results_{timestamp}.csv", index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   - test_results_{timestamp}.json")
        print(f"   - test_results_{timestamp}.csv")

def main():
    """Main testing function"""
    print("🚀 Elite Semantic Cache Testing System")
    print("=" * 80)
    print("Testing with 100+ prompts designed to expose false positives...")
    
    # Initialize tester
    tester = EliteCacheTester()
    
    # Check if cache system is running
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        print("✅ Cache system is running")
    except:
        print("❌ Cache system is not running!")
        print("Please start the system: python app.py")
        return
    
    # Choose test mode
    print("\nChoose test mode:")
    print("1. Sequential Testing (slower, easier on system)")
    print("2. Parallel Testing (faster, stress test)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return
    
    start_time = time.time()
    
    if choice == "1":
        results = tester.run_sequential_test()
    elif choice == "2":
        results = tester.run_parallel_test()
    else:
        print("Invalid choice, using sequential mode...")
        results = tester.run_sequential_test()
    
    end_time = time.time()
    
    # Analyze results
    analysis = tester.analyze_results(results)
    analysis['total_test_time'] = round(end_time - start_time, 2)
    
    # Generate report
    tester.generate_report(results, analysis)
    
    # Save results
    tester.save_results(results, analysis)
    
    print(f"\n✅ Testing completed in {analysis['total_test_time']}s")
    print("🎯 Check the generated files for detailed analysis!")

if __name__ == "__main__":
    try:
        # Install required packages if not available
        try:
            import pandas as pd
        except ImportError:
            print("Installing pandas...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
            import pandas as pd
        
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please ensure the semantic cache system is running on http://localhost:5000")