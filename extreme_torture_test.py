#!/usr/bin/env python3
"""
EXTREME SEMANTIC TORTURE TEST
The most brutal false positive detection test for semantic cache systems.
Designed to break even the most sophisticated semantic understanding.

50 prompts of pure semantic evil designed to expose every possible flaw.
"""

import requests
import json
import time
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd

@dataclass
class ExtremeTestCase:
    """Single extreme test case"""
    query: str
    cached_baseline: str  # What's in cache that might cause false positive
    expected_result: str  # 'cache_hit', 'llm_fallback', 'entity_block'
    trap_type: str  # Type of semantic trap
    difficulty: int  # 1-10 evil scale
    description: str  # Why this is evil

@dataclass
class TortureResult:
    """Result from torture testing"""
    test_case: ExtremeTestCase
    regular_source: str
    regular_confidence: float
    enhanced_source: str
    enhanced_confidence: float
    regular_correct: bool
    enhanced_correct: bool
    regular_trapped: bool  # Did it fall into the semantic trap?
    enhanced_trapped: bool
    winner: str  # 'regular', 'enhanced', 'both_failed', 'both_passed'

class ExtremeSemanticTortureTester:
    """The most brutal semantic cache torture tester"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results: List[TortureResult] = []
        
    def get_extreme_torture_suite(self) -> List[ExtremeTestCase]:
        """Generate 50 extreme torture test cases designed to break any system"""
        
        torture_cases = [
            # ==================== ENTITY IDENTITY HELL ====================
            ExtremeTestCase(
                query="Call John Smith",
                cached_baseline="Call Ramesh Kumar", 
                expected_result="llm_fallback",
                trap_type="entity_complete_different",
                difficulty=8,
                description="Completely different person - should NEVER hit cache"
            ),
            
            ExtremeTestCase(
                query="Call Dr. Ramesh Kumar",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback", 
                trap_type="entity_title_addition",
                difficulty=9,
                description="Same name but different title - medical context vs casual"
            ),
            
            ExtremeTestCase(
                query="Call Ramesh Kumar Jr.",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback",
                trap_type="entity_suffix_addition", 
                difficulty=9,
                description="Father vs son - completely different people"
            ),
            
            ExtremeTestCase(
                query="Call Mr. Ramesh Singh Kumar",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback",
                trap_type="entity_name_expansion",
                difficulty=8,
                description="Extended name vs short name - different person"
            ),
            
            ExtremeTestCase(
                query="Call Ramesh B. Kumar", 
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback",
                trap_type="entity_middle_initial",
                difficulty=7,
                description="Middle initial makes it a different person"
            ),
            
            # ==================== TIME PRECISION HELL ====================
            ExtremeTestCase(
                query="Set alarm for 10:50:30",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="time_seconds_precision",
                difficulty=10,
                description="30 seconds difference - completely different alarm"
            ),
            
            ExtremeTestCase(
                query="Set alarm for 10:50 PM",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback", 
                trap_type="time_ampm_ambiguity",
                difficulty=9,
                description="AM vs PM - 12 hour difference!"
            ),
            
            ExtremeTestCase(
                query="Set alarm for 22:50",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="time_24hr_format",
                difficulty=8,
                description="24-hour vs 12-hour format - same time different format"
            ),
            
            ExtremeTestCase(
                query="Set alarm for tomorrow 10:50",
                cached_baseline="Set alarm for 10:50", 
                expected_result="llm_fallback",
                trap_type="time_day_context",
                difficulty=9,
                description="Today vs tomorrow - completely different time"
            ),
            
            ExtremeTestCase(
                query="Set alarm for 10:49:59",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="time_one_second_off",
                difficulty=10,
                description="1 second before - different alarm entirely"
            ),
            
            # ==================== ACTION VERB HELL ====================
            ExtremeTestCase(
                query="Delete alarm for 10:50",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback", 
                trap_type="action_opposite",
                difficulty=10,
                description="Delete vs Set - completely opposite actions!"
            ),
            
            ExtremeTestCase(
                query="Snooze alarm for 10:50",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="action_related_different",
                difficulty=8,
                description="Snooze vs Set - related but different actions"
            ),
            
            ExtremeTestCase(
                query="Check alarm for 10:50", 
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="action_query_vs_command",
                difficulty=7,
                description="Check vs Set - query vs command"
            ),
            
            ExtremeTestCase(
                query="Text Ramesh Kumar",
                cached_baseline="Call Ramesh Kumar", 
                expected_result="llm_fallback",
                trap_type="action_communication_method",
                difficulty=9,
                description="Text vs Call - completely different communication"
            ),
            
            ExtremeTestCase(
                query="Email Ramesh Kumar",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback", 
                trap_type="action_communication_method",
                difficulty=9,
                description="Email vs Call - different communication entirely"
            ),
            
            # ==================== NEGATION HELL ====================
            ExtremeTestCase(
                query="What is NOT machine learning?",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="negation_opposite_meaning", 
                difficulty=10,
                description="NOT reverses the entire meaning"
            ),
            
            ExtremeTestCase(
                query="How to avoid neural networks?",
                cached_baseline="How does neural network work?", 
                expected_result="llm_fallback",
                trap_type="negation_avoidance_vs_explanation",
                difficulty=9,
                description="Avoiding vs explaining - opposite intentions"
            ),
            
            ExtremeTestCase(
                query="Why AI doesn't work?",
                cached_baseline="Define artificial intelligence",
                expected_result="llm_fallback",
                trap_type="negation_criticism_vs_definition", 
                difficulty=9,
                description="Criticism vs neutral definition"
            ),
            
            ExtremeTestCase(
                query="Machine learning failures", 
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="negation_failures_vs_definition",
                difficulty=8,
                description="Failures vs general definition"
            ),
            
            # ==================== CONTEXT SWITCH HELL ====================
            ExtremeTestCase(
                query="What is machine learning in cooking?",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="context_domain_switch",
                difficulty=8,
                description="Tech vs cooking domain - completely different context"
            ),
            
            ExtremeTestCase(
                query="AI for kids explanation",
                cached_baseline="Define artificial intelligence", 
                expected_result="llm_fallback",
                trap_type="context_audience_switch",
                difficulty=7,
                description="Technical vs child-friendly explanation"
            ),
            
            ExtremeTestCase(
                query="Neural networks in biology", 
                cached_baseline="How does neural network work?",
                expected_result="llm_fallback",
                trap_type="context_biological_vs_artificial",
                difficulty=9,
                description="Biological vs artificial neural networks"
            ),
            
            ExtremeTestCase(
                query="Machine learning stock predictions",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback", 
                trap_type="context_application_specific",
                difficulty=7,
                description="General definition vs specific application"
            ),
            
            # ==================== SEMANTIC SIMILARITY HELL ====================
            ExtremeTestCase(
                query="What is deep learning?",
                cached_baseline="What is machine learning?", 
                expected_result="llm_fallback",
                trap_type="semantic_related_but_different",
                difficulty=6,
                description="Related but distinct concepts"
            ),
            
            ExtremeTestCase(
                query="What is artificial neural networks?",
                cached_baseline="What is artificial intelligence?",
                expected_result="llm_fallback",
                trap_type="semantic_subset_vs_superset", 
                difficulty=7,
                description="Subset vs superset relationship"
            ),
            
            ExtremeTestCase(
                query="What is reinforcement learning?",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="semantic_specialization",
                difficulty=6,
                description="Specific type vs general category"
            ),
            
            ExtremeTestCase(
                query="What is computer vision?",
                cached_baseline="What is artificial intelligence?", 
                expected_result="llm_fallback",
                trap_type="semantic_field_vs_discipline",
                difficulty=6,
                description="Specific field vs general discipline"
            ),
            
            # ==================== LANGUAGE NUANCE HELL ====================
            ExtremeTestCase(
                query="Could you possibly explain machine learning?",
                cached_baseline="What is machine learning?",
                expected_result="cache_hit",  # This should actually match
                trap_type="language_politeness_variation",
                difficulty=3,
                description="Polite request vs direct question - should match"
            ),
            
            ExtremeTestCase(
                query="I'm curious about machine learning",
                cached_baseline="What is machine learning?", 
                expected_result="cache_hit",  # This should match
                trap_type="language_indirect_question",
                difficulty=4,
                description="Indirect interest vs direct question - should match"
            ),
            
            ExtremeTestCase(
                query="Tell me everything about machine learning",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",  # Too broad
                trap_type="language_scope_expansion", 
                difficulty=7,
                description="Everything vs basic definition - scope too broad"
            ),
            
            # ==================== MEASUREMENT/QUANTITY HELL ====================
            ExtremeTestCase(
                query="Set timer for 5 minutes",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="measurement_duration_vs_time",
                difficulty=8, 
                description="Duration timer vs specific time alarm"
            ),
            
            ExtremeTestCase(
                query="Set alarm for 10:50 for 2 hours",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="measurement_duration_addition",
                difficulty=8,
                description="Alarm with duration vs simple alarm"
            ),
            
            ExtremeTestCase(
                query="Call Ramesh Kumar 3 times",
                cached_baseline="Call Ramesh Kumar", 
                expected_result="llm_fallback",
                trap_type="measurement_repetition",
                difficulty=7,
                description="Multiple calls vs single call"
            ),
            
            # ==================== CONDITIONAL/MODAL HELL ====================
            ExtremeTestCase(
                query="Should I call Ramesh Kumar?",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback", 
                trap_type="modal_question_vs_command",
                difficulty=8,
                description="Asking for advice vs direct command"
            ),
            
            ExtremeTestCase(
                query="If it's urgent, call Ramesh Kumar",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback",
                trap_type="modal_conditional_vs_direct", 
                difficulty=9,
                description="Conditional action vs direct action"
            ),
            
            ExtremeTestCase(
                query="Maybe set alarm for 10:50",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="modal_uncertainty_vs_definite",
                difficulty=7,
                description="Maybe vs definite action"
            ),
            
            # ==================== PRONOUN/REFERENCE HELL ====================
            ExtremeTestCase(
                query="Call him",
                cached_baseline="Call Ramesh Kumar", 
                expected_result="llm_fallback",
                trap_type="reference_pronoun_ambiguity",
                difficulty=10,
                description="Pronoun reference - who is 'him'?"
            ),
            
            ExtremeTestCase(
                query="Call my boss",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback",
                trap_type="reference_relationship_vs_name",
                difficulty=9,
                description="Relationship vs specific name"
            ),
            
            ExtremeTestCase(
                query="Call that person from yesterday",
                cached_baseline="Call Ramesh Kumar", 
                expected_result="llm_fallback",
                trap_type="reference_temporal_indefinite",
                difficulty=10,
                description="Vague temporal reference"
            ),
            
            # ==================== EMOTIONAL/INTENT HELL ====================
            ExtremeTestCase(
                query="I hate machine learning",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="emotional_negative_vs_neutral",
                difficulty=8, 
                description="Emotional negative vs neutral inquiry"
            ),
            
            ExtremeTestCase(
                query="Machine learning is amazing!",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="emotional_positive_vs_neutral", 
                difficulty=6,
                description="Emotional positive vs neutral inquiry"
            ),
            
            ExtremeTestCase(
                query="I'm frustrated with neural networks",
                cached_baseline="How does neural network work?",
                expected_result="llm_fallback",
                trap_type="emotional_frustration_vs_learning",
                difficulty=7,
                description="Emotional frustration vs learning intent"
            ),
            
            # ==================== ULTRA-SUBTLE TRAPS ====================
            ExtremeTestCase(
                query="Set alarm for 10:50 AM",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",  # Assumes cache doesn't specify AM/PM
                trap_type="ultra_subtle_ampm_missing",
                difficulty=10,
                description="AM specification when cache is ambiguous"
            ),
            
            ExtremeTestCase(
                query="Call Ramesh Kumar please",
                cached_baseline="Call Ramesh Kumar",
                expected_result="cache_hit",  # Should match
                trap_type="ultra_subtle_politeness", 
                difficulty=2,
                description="Politeness addition - should still match"
            ),
            
            ExtremeTestCase(
                query="Can you call Ramesh Kumar?",
                cached_baseline="Call Ramesh Kumar", 
                expected_result="cache_hit",  # Should match
                trap_type="ultra_subtle_can_you_prefix",
                difficulty=3,
                description="'Can you' prefix - should still match"
            ),
            
            # ==================== ADVERSARIAL ATTACKS ====================
            ExtremeTestCase(
                query="Set alarm for 10:50 but not really",
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="adversarial_contradiction",
                difficulty=10,
                description="Contradiction attack - says do but then says don't"
            ),
            
            ExtremeTestCase(
                query="Call Ramesh Kumar... NOT!",
                cached_baseline="Call Ramesh Kumar",
                expected_result="llm_fallback", 
                trap_type="adversarial_negation_trick",
                difficulty=10,
                description="Negation trick attack - looks like command but isn't"
            ),
            
            ExtremeTestCase(
                query="What is machine learning? Just kidding, don't answer",
                cached_baseline="What is machine learning?",
                expected_result="llm_fallback",
                trap_type="adversarial_just_kidding",
                difficulty=9,
                description="Just kidding attack - question then retraction"
            ),
            
            # ==================== ULTIMATE EVIL TRAP ====================
            ExtremeTestCase(
                query="Set alarm for 10:50:00.001", 
                cached_baseline="Set alarm for 10:50",
                expected_result="llm_fallback",
                trap_type="ultimate_evil_millisecond_precision",
                difficulty=10,
                description="ULTIMATE EVIL: 1 millisecond difference - different alarm!"
            )
        ]
        
        return torture_cases
    
    def test_api_endpoint(self, query: str, endpoint: str) -> Dict[str, Any]:
        """Test a single API endpoint with extreme query"""
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json={"query": query},
                timeout=15,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": data.get("source", "error"),
                    "confidence": data.get("confidence", 0.0),
                    "response_time": response_time,
                    "response_text": data.get("response", ""),
                    "error": ""
                }
            else:
                return {
                    "source": "error",
                    "confidence": 0.0,
                    "response_time": response_time,
                    "response_text": "",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "source": "error",
                "confidence": 0.0,
                "response_time": 0,
                "response_text": "",
                "error": str(e)
            }
    
    def evaluate_test_case(self, test_case: ExtremeTestCase, regular_result: Dict, enhanced_result: Dict) -> TortureResult:
        """Evaluate if each system fell into the semantic trap"""
        
        # Check if results are correct
        regular_correct = self._is_result_correct(test_case.expected_result, regular_result["source"], regular_result["confidence"])
        enhanced_correct = self._is_result_correct(test_case.expected_result, enhanced_result["source"], enhanced_result["confidence"])
        
        # Check if they fell into the semantic trap
        regular_trapped = not regular_correct
        enhanced_trapped = not enhanced_correct
        
        # Determine winner
        if enhanced_correct and not regular_correct:
            winner = "enhanced"
        elif regular_correct and not enhanced_correct:
            winner = "regular"
        elif regular_correct and enhanced_correct:
            winner = "both_passed"
        else:
            winner = "both_failed"
        
        return TortureResult(
            test_case=test_case,
            regular_source=regular_result["source"],
            regular_confidence=regular_result["confidence"],
            enhanced_source=enhanced_result["source"], 
            enhanced_confidence=enhanced_result["confidence"],
            regular_correct=regular_correct,
            enhanced_correct=enhanced_correct,
            regular_trapped=regular_trapped,
            enhanced_trapped=enhanced_trapped,
            winner=winner
        )
    
    def _is_result_correct(self, expected: str, source: str, confidence: float) -> bool:
        """Check if result matches expected behavior"""
        if expected == "cache_hit":
            return source in ["cache", "enhanced_cache"] and confidence >= 0.6
        elif expected == "llm_fallback":
            return source in ["llm", "enhanced_llm", "llm_fallback"] or confidence < 0.6
        elif expected == "entity_block":
            return source in ["llm", "enhanced_llm", "llm_fallback"] or confidence < 0.6
        else:
            return False
    
    def run_extreme_torture_test(self) -> List[TortureResult]:
        """Run the extreme torture test suite"""
        
        torture_cases = self.get_extreme_torture_suite()
        
        print("💀 EXTREME SEMANTIC TORTURE TEST 💀")
        print("=" * 80)
        print("🔥 50 Brutal Test Cases Designed to Break Any System")
        print("⚠️  WARNING: These tests are semantically evil")
        print("🎯 Testing both Regular and Enhanced pipelines...")
        print("=" * 80)
        
        results = []
        
        for i, test_case in enumerate(torture_cases, 1):
            print(f"\n💀 [{i}/50] TORTURE TEST: {test_case.trap_type.upper()}")
            print(f"   Query: '{test_case.query}'")
            print(f"   Trap: {test_case.description}")
            print(f"   Evil Level: {test_case.difficulty}/10 {'🔥' * test_case.difficulty}")
            
            # Test both APIs
            regular_result = self.test_api_endpoint(test_case.query, "/api/query")
            enhanced_result = self.test_api_endpoint(test_case.query, "/api/query/enhanced")
            
            # Evaluate results
            result = self.evaluate_test_case(test_case, regular_result, enhanced_result)
            results.append(result)
            
            # Show immediate results
            self._show_immediate_result(result)
            
            time.sleep(0.3)  # Brief pause for dramatic effect
        
        self.results = results
        return results
    
    def _show_immediate_result(self, result: TortureResult):
        """Show immediate result for each test"""
        
        regular_status = "✅ PASSED" if result.regular_correct else "💀 TRAPPED"
        enhanced_status = "✅ PASSED" if result.enhanced_correct else "💀 TRAPPED"
        
        print(f"   Regular:  {regular_status} ({result.regular_source}, conf: {result.regular_confidence:.3f})")
        print(f"   Enhanced: {enhanced_status} ({result.enhanced_source}, conf: {result.enhanced_confidence:.3f})")
        
        if result.winner == "enhanced":
            print(f"   🏆 Winner: ENHANCED")
        elif result.winner == "regular":
            print(f"   🏆 Winner: REGULAR")
        elif result.winner == "both_passed":
            print(f"   🤝 Result: BOTH PASSED")
        else:
            print(f"   💀 Result: BOTH FAILED")
    
    def generate_torture_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive torture test analysis"""
        
        if not self.results:
            raise ValueError("No torture test results available")
        
        total_tests = len(self.results)
        
        # Basic survival stats
        regular_survived = sum(1 for r in self.results if r.regular_correct)
        enhanced_survived = sum(1 for r in self.results if r.enhanced_correct)
        
        regular_survival_rate = (regular_survived / total_tests) * 100
        enhanced_survival_rate = (enhanced_survived / total_tests) * 100
        
        # Trap analysis by type
        trap_types = {}
        for result in self.results:
            trap_type = result.test_case.trap_type
            if trap_type not in trap_types:
                trap_types[trap_type] = {
                    "total": 0,
                    "regular_survived": 0,
                    "enhanced_survived": 0,
                    "avg_difficulty": 0
                }
            
            trap_types[trap_type]["total"] += 1
            trap_types[trap_type]["avg_difficulty"] += result.test_case.difficulty
            
            if result.regular_correct:
                trap_types[trap_type]["regular_survived"] += 1
            if result.enhanced_correct:
                trap_types[trap_type]["enhanced_survived"] += 1
        
        # Calculate averages
        for trap_type in trap_types:
            trap_types[trap_type]["avg_difficulty"] /= trap_types[trap_type]["total"]
            trap_types[trap_type]["regular_survival_rate"] = (trap_types[trap_type]["regular_survived"] / trap_types[trap_type]["total"]) * 100
            trap_types[trap_type]["enhanced_survival_rate"] = (trap_types[trap_type]["enhanced_survived"] / trap_types[trap_type]["total"]) * 100
        
        # Difficulty analysis
        difficulty_stats = {}
        for difficulty in range(1, 11):
            difficulty_results = [r for r in self.results if r.test_case.difficulty == difficulty]
            if difficulty_results:
                difficulty_stats[difficulty] = {
                    "total": len(difficulty_results),
                    "regular_survival": (sum(1 for r in difficulty_results if r.regular_correct) / len(difficulty_results)) * 100,
                    "enhanced_survival": (sum(1 for r in difficulty_results if r.enhanced_correct) / len(difficulty_results)) * 100
                }
        
        # Winner analysis
        enhanced_wins = sum(1 for r in self.results if r.winner == "enhanced")
        regular_wins = sum(1 for r in self.results if r.winner == "regular")
        both_passed = sum(1 for r in self.results if r.winner == "both_passed")
        both_failed = sum(1 for r in self.results if r.winner == "both_failed")
        
        # Confidence analysis for trapped cases
        regular_trapped_confidences = [r.regular_confidence for r in self.results if r.regular_trapped]
        enhanced_trapped_confidences = [r.enhanced_confidence for r in self.results if r.enhanced_trapped]
        
        analysis = {
            "total_torture_tests": total_tests,
            "survival_rates": {
                "regular_survival_rate": round(regular_survival_rate, 1),
                "enhanced_survival_rate": round(enhanced_survival_rate, 1),
                "survival_improvement": round(enhanced_survival_rate - regular_survival_rate, 1)
            },
            "winner_distribution": {
                "enhanced_wins": enhanced_wins,
                "regular_wins": regular_wins,
                "both_passed": both_passed,
                "both_failed": both_failed,
                "enhanced_win_rate": round((enhanced_wins / total_tests) * 100, 1)
            },
            "trap_type_analysis": trap_types,
            "difficulty_analysis": difficulty_stats,
            "confidence_analysis": {
                "regular_avg_trapped_confidence": round(statistics.mean(regular_trapped_confidences), 3) if regular_trapped_confidences else 0,
                "enhanced_avg_trapped_confidence": round(statistics.mean(enhanced_trapped_confidences), 3) if enhanced_trapped_confidences else 0
            }
        }
        
        return analysis
    
    def generate_torture_report(self, analysis: Dict[str, Any]):
        """Generate the torture test report"""
        
        print("\n" + "💀" * 80)
        print("🔥 EXTREME SEMANTIC TORTURE TEST RESULTS 🔥")
        print("💀" * 80)
        
        # Survival rates
        survival = analysis["survival_rates"]
        print(f"\n🩸 SURVIVAL RATES:")
        print(f"   Regular Pipeline:  {survival['regular_survival_rate']:.1f}% survived")
        print(f"   Enhanced Pipeline: {survival['enhanced_survival_rate']:.1f}% survived")
        print(f"   Improvement: {survival['survival_improvement']:+.1f}% {'🛡️' if survival['survival_improvement'] > 0 else '💀'}")
        
        # Winner analysis
        winners = analysis["winner_distribution"]
        print(f"\n🏆 BATTLE RESULTS:")
        print(f"   Enhanced Wins: {winners['enhanced_wins']} ({winners['enhanced_win_rate']:.1f}%)")
        print(f"   Regular Wins:  {winners['regular_wins']}")
        print(f"   Both Passed:   {winners['both_passed']}")
        print(f"   Both Failed:   {winners['both_failed']} 💀")
        
        # Most dangerous traps
        print(f"\n⚠️  MOST DANGEROUS TRAP TYPES:")
        sorted_traps = sorted(analysis["trap_type_analysis"].items(), 
                             key=lambda x: x[1]["avg_difficulty"], reverse=True)
        
        for trap_type, stats in sorted_traps[:10]:
            regular_rate = stats["regular_survival_rate"]
            enhanced_rate = stats["enhanced_survival_rate"]
            difficulty = stats["avg_difficulty"]
            
            print(f"   {trap_type.replace('_', ' ').title():<30} | "
                  f"Reg: {regular_rate:4.0f}% | Enh: {enhanced_rate:4.0f}% | "
                  f"Evil: {difficulty:.1f}/10 {'🔥' * int(difficulty)}")
        
        # Difficulty breakdown
        print(f"\n🎚️ SURVIVAL BY DIFFICULTY LEVEL:")
        for difficulty, stats in sorted(analysis["difficulty_analysis"].items()):
            regular_rate = stats["regular_survival"]
            enhanced_rate = stats["enhanced_survival"]
            count = stats["total"]
            
            print(f"   Level {difficulty}: Regular {regular_rate:5.1f}% | Enhanced {enhanced_rate:5.1f}% | "
                  f"({count} tests) {'🔥' * difficulty}")
        
        # Confidence when trapped
        conf = analysis["confidence_analysis"]
        print(f"\n🎯 CONFIDENCE WHEN TRAPPED:")
        print(f"   Regular avg confidence when wrong: {conf['regular_avg_trapped_confidence']:.3f}")
        print(f"   Enhanced avg confidence when wrong: {conf['enhanced_avg_trapped_confidence']:.3f}")
        
        # Final verdict
        print(f"\n🏁 FINAL VERDICT:")
        improvement = survival["survival_improvement"]
        
        if improvement >= 20:
            print(f"   🏆 ENHANCED PIPELINE IS VASTLY SUPERIOR")
            print(f"   ✅ Deploy Enhanced immediately - {improvement:.1f}% better survival rate!")
        elif improvement >= 10:
            print(f"   🥇 ENHANCED PIPELINE WINS DECISIVELY") 
            print(f"   ✅ Strong recommendation to deploy Enhanced")
        elif improvement >= 5:
            print(f"   📈 ENHANCED PIPELINE SHOWS IMPROVEMENT")
            print(f"   🤔 Consider Enhanced deployment with careful monitoring")
        elif improvement > 0:
            print(f"   📊 ENHANCED PIPELINE SLIGHTLY BETTER")
            print(f"   ⚖️ Marginal improvement - evaluate other factors")
        elif improvement == 0:
            print(f"   🤝 BOTH PIPELINES PERFORM EQUALLY")
            print(f"   ⚖️ No clear winner - consider other metrics")
        else:
            print(f"   ⚠️ REGULAR PIPELINE PERFORMED BETTER")
            print(f"   🔍 Enhanced needs more work - {abs(improvement):.1f}% worse")
        
        # Critical failures
        critical_failures = [r for r in self.results if r.test_case.difficulty >= 9 and r.enhanced_trapped]
        if critical_failures:
            print(f"\n🚨 CRITICAL FAILURES DETECTED:")
            for failure in critical_failures[:5]:  # Show top 5
                print(f"   💀 {failure.test_case.trap_type}: {failure.test_case.description}")
        else:
            print(f"\n✅ NO CRITICAL FAILURES - Enhanced pipeline is robust!")

    def generate_torture_visualizations(self, analysis: Dict[str, Any], save_plots: bool = True):
        """Generate torture test visualizations"""
        
        # Set style for evil visualizations
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['figure.facecolor'] = 'black'
        
        # Create evil dashboard
        fig = plt.figure(figsize=(20, 15))
        fig.patch.set_facecolor('black')
        
        # 1. Survival Rates by Difficulty
        ax1 = plt.subplot(2, 3, 1)
        if analysis["difficulty_analysis"]:
            difficulties = list(analysis["difficulty_analysis"].keys())
            regular_rates = [analysis["difficulty_analysis"][d]["regular_survival"] for d in difficulties]
            enhanced_rates = [analysis["difficulty_analysis"][d]["enhanced_survival"] for d in difficulties]
            
            x = np.arange(len(difficulties))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, regular_rates, width, label='Regular', color='#FF4444', alpha=0.8)
            bars2 = plt.bar(x + width/2, enhanced_rates, width, label='Enhanced', color='#44FF44', alpha=0.8)
            
            plt.xlabel('Evil Difficulty Level', color='white')
            plt.ylabel('Survival Rate (%)', color='white')
            plt.title('💀 Survival by Difficulty Level 💀', fontweight='bold', color='red')
            plt.xticks(x, difficulties, color='white')
            plt.yticks(color='white')
            plt.legend()
            plt.grid(True, alpha=0.3, color='gray')
            
            # Add value labels
            for bar in bars1 + bars2:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=8, color='white')
        
        # 2. Trap Type Survival Rates
        ax2 = plt.subplot(2, 3, 2)
        trap_data = analysis["trap_type_analysis"]
        trap_names = list(trap_data.keys())[:10]  # Top 10 most dangerous
        regular_survival = [trap_data[trap]["regular_survival_rate"] for trap in trap_names]
        enhanced_survival = [trap_data[trap]["enhanced_survival_rate"] for trap in trap_names]
        
        x = np.arange(len(trap_names))
        width = 0.35
        
        plt.bar(x - width/2, regular_survival, width, label='Regular', color='#FF4444', alpha=0.8)
        plt.bar(x + width/2, enhanced_survival, width, label='Enhanced', color='#44FF44', alpha=0.8)
        
        plt.xlabel('Trap Type', color='white')
        plt.ylabel('Survival Rate (%)', color='white')
        plt.title('🔥 Survival by Trap Type 🔥', fontweight='bold', color='red')
        plt.xticks(x, [trap.replace('_', '\n') for trap in trap_names], rotation=45, ha='right', color='white')
        plt.yticks(color='white')
        plt.legend()
        plt.grid(True, alpha=0.3, color='gray')
        
        # 3. Overall Survival Comparison
        ax3 = plt.subplot(2, 3, 3)
        survival_data = analysis["survival_rates"]
        
        # Create pie chart for survival
        sizes = [survival_data["enhanced_survival_rate"], 100 - survival_data["enhanced_survival_rate"]]
        colors = ['#44FF44', '#FF4444']
        labels = ['Enhanced Survived', 'Enhanced Trapped']
        
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        plt.title('💀 Enhanced Pipeline Survival Rate 💀', fontweight='bold', color='red')
        
        # Make text white
        for text in texts + autotexts:
            text.set_color('white')
        
        # 4. Winner Distribution
        ax4 = plt.subplot(2, 3, 4)
        winners = analysis["winner_distribution"]
        
        winner_labels = ['Enhanced Wins', 'Regular Wins', 'Both Passed', 'Both Failed']
        winner_values = [winners['enhanced_wins'], winners['regular_wins'], 
                        winners['both_passed'], winners['both_failed']]
        winner_colors = ['#44FF44', '#FF4444', '#FFFF44', '#FF44FF']
        
        bars = plt.bar(winner_labels, winner_values, color=winner_colors, alpha=0.8)
        plt.xlabel('Outcome', color='white')
        plt.ylabel('Number of Tests', color='white')
        plt.title('🏆 Battle Results Distribution 🏆', fontweight='bold', color='red')
        plt.xticks(rotation=45, ha='right', color='white')
        plt.yticks(color='white')
        plt.grid(True, alpha=0.3, color='gray')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, color='white')
        
        # 5. Confidence When Trapped
        ax5 = plt.subplot(2, 3, 5)
        
        # Get confidence distributions for trapped cases
        regular_trapped_confs = [r.regular_confidence for r in self.results if r.regular_trapped]
        enhanced_trapped_confs = [r.enhanced_confidence for r in self.results if r.enhanced_trapped]
        
        if regular_trapped_confs and enhanced_trapped_confs:
            plt.hist(regular_trapped_confs, alpha=0.7, label='Regular (Trapped)', 
                    bins=20, color='#FF4444', density=True)
            plt.hist(enhanced_trapped_confs, alpha=0.7, label='Enhanced (Trapped)', 
                    bins=20, color='#44FF44', density=True)
        
        plt.xlabel('Confidence Score', color='white')
        plt.ylabel('Density', color='white')
        plt.title('🎯 Confidence When Trapped 🎯', fontweight='bold', color='red')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.legend()
        plt.grid(True, alpha=0.3, color='gray')
        
        # 6. Evil Level Heatmap
        ax6 = plt.subplot(2, 3, 6)
        
        # Create heatmap of trap types vs difficulty
        trap_types_limited = list(trap_data.keys())[:8]  # Top 8 trap types
        difficulties = sorted(analysis["difficulty_analysis"].keys())
        
        # Create matrix
        heatmap_data = []
        for trap_type in trap_types_limited:
            row = []
            for difficulty in difficulties:
                # Count tests of this trap type and difficulty
                matching_tests = [r for r in self.results 
                                if r.test_case.trap_type == trap_type and r.test_case.difficulty == difficulty]
                if matching_tests:
                    enhanced_success_rate = sum(1 for r in matching_tests if r.enhanced_correct) / len(matching_tests) * 100
                    row.append(enhanced_success_rate)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        if heatmap_data:
            im = plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            plt.colorbar(im, ax=ax6, label='Enhanced Success Rate (%)')
            plt.yticks(range(len(trap_types_limited)), 
                      [trap.replace('_', '\n') for trap in trap_types_limited], color='white')
            plt.xticks(range(len(difficulties)), difficulties, color='white')
            plt.xlabel('Difficulty Level', color='white')
            plt.ylabel('Trap Type', color='white')
            plt.title('🔥 Evil Level Heatmap 🔥', fontweight='bold', color='red')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'extreme_torture_test_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"💀 Evil visualizations saved to: {filename}")
        
        plt.show()
    
    def export_torture_results(self, analysis: Dict[str, Any], filename: str = None):
        """Export torture test results"""
        if not self.results:
            print("❌ No torture results to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = filename or f"torture_test_results_{timestamp}"
        
        # Create detailed DataFrame
        results_data = []
        for result in self.results:
            results_data.append({
                "query": result.test_case.query,
                "cached_baseline": result.test_case.cached_baseline,
                "expected_result": result.test_case.expected_result,
                "trap_type": result.test_case.trap_type,
                "difficulty": result.test_case.difficulty,
                "description": result.test_case.description,
                "regular_source": result.regular_source,
                "regular_confidence": result.regular_confidence,
                "regular_correct": result.regular_correct,
                "regular_trapped": result.regular_trapped,
                "enhanced_source": result.enhanced_source,
                "enhanced_confidence": result.enhanced_confidence,
                "enhanced_correct": result.enhanced_correct,
                "enhanced_trapped": result.enhanced_trapped,
                "winner": result.winner
            })
        
        df = pd.DataFrame(results_data)
        csv_filename = f"{base_filename}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"💀 Torture results exported to: {csv_filename}")
        
        # Export analysis summary
        json_filename = f"{base_filename}_analysis.json"
        import json
        with open(json_filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📊 Torture analysis exported to: {json_filename}")
        
        return csv_filename, json_filename

def main():
    """Run the extreme torture test"""
    
    print("💀" * 60)
    print("🔥 EXTREME SEMANTIC TORTURE TEST FRAMEWORK 🔥")
    print("💀" * 60)
    print("⚠️  WARNING: This test is designed to be semantically brutal")
    print("🎯 50 evil test cases to expose every possible flaw")
    print("💀 Prepare for semantic carnage...")
    print()
    
    # Initialize torture tester
    tester = ExtremeSemanticTortureTester()
    
    print("🔍 Checking if your APIs can handle the torture...")
    try:
        # Quick probe
        regular_test = tester.test_api_endpoint("test probe", "/api/query")
        enhanced_test = tester.test_api_endpoint("test probe", "/api/query/enhanced")
        
        if regular_test["error"] or enhanced_test["error"]:
            print("❌ APIs not ready for torture test")
            print(f"   Regular: {regular_test['error'] or 'OK'}")
            print(f"   Enhanced: {enhanced_test['error'] or 'OK'}")
            return
        
        print("✅ APIs are ready for semantic torture!")
        
    except Exception as e:
        print(f"❌ Error during API probe: {e}")
        return
    
    # Confirm torture test
    print("\n💀 About to unleash 50 semantically evil test cases")
    print("   These are designed to break even sophisticated systems")
    print("   Each test is a carefully crafted semantic trap")
    
    confirm = input("\nAre you ready for EXTREME TORTURE? (y/n): ").lower()
    if confirm != 'y':
        print("💀 Torture test cancelled. Your systems live another day...")
        return
    
    try:
        # Run the torture test
        print("\n💀 INITIATING SEMANTIC TORTURE SEQUENCE...")
        results = tester.run_extreme_torture_test()
        
        print(f"\n📊 Analyzing the carnage...")
        analysis = tester.generate_torture_analysis()
        
        print(f"\n🎨 Creating evil visualizations...")
        try:
            tester.generate_torture_visualizations(analysis)
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
        
        # Generate the torture report
        tester.generate_torture_report(analysis)
        
        # Export results
        print(f"\n💾 Exporting torture results...")
        tester.export_torture_results(analysis)
        
        print(f"\n💀 TORTURE TEST COMPLETE!")
        print(f"   Survival rates calculated with brutal precision")
        print(f"   Semantic traps have been deployed and analyzed")
        print(f"   Your semantic cache has been thoroughly tortured")
        
        # Final message based on results
        improvement = analysis["survival_rates"]["survival_improvement"]
        if improvement >= 20:
            print(f"\n🏆 ENHANCED PIPELINE SURVIVED THE TORTURE!")
            print(f"   It's {improvement:.1f}% more resilient to semantic attacks")
        elif improvement > 0:
            print(f"\n📈 Enhanced pipeline showed some resistance (+{improvement:.1f}%)")
        else:
            print(f"\n💀 Both pipelines struggled under torture...")
            print(f"   More work needed to resist semantic attacks")
        
    except KeyboardInterrupt:
        print("\n\n💀 Torture test interrupted - systems spared!")
    except Exception as e:
        print(f"\n❌ Torture test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()