"""
Confidence Aggregation Logic
Combines multiple signals to determine cache hit confidence
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class ConfidenceAggregator:
    """Aggregates multiple confidence signals for cache decisions"""
    
    def __init__(self):
        """Initialize confidence aggregator with thresholds"""
        self.thresholds = {
            "high_confidence": 0.85,    # Use cache
            "medium_confidence": 0.6,   # Optional LLM call
            "low_confidence": 0.0       # Fallback to LLM
        }
        
        self.weights = {
            "similarity": 0.3,      # Reduced cosine similarity weight
            "reranker": 0.25,       # Reduced cross-encoder reranker weight
            "intent": 0.15,         # Intent classification weight
            "entity": 0.3          # INCREASED entity matching weight (critical!)
        }
    
    def compute_confidence(self, 
                          similarity_score: float,
                          reranker_score: float,
                          intent_similarity: float,
                          entity_match_score: float = 0.0,
                          requires_precise_matching: bool = False,
                          **kwargs) -> Dict[str, Any]:
        """
        Compute aggregated confidence score with enhanced entity checking
        
        Args:
            similarity_score: Cosine similarity score (0-1)
            reranker_score: Cross-encoder relevance score (0-1)
            intent_similarity: Intent classification similarity (0-1)
            entity_match_score: Entity matching score (0-1)
            requires_precise_matching: Whether this query requires precise entity matching
            
        Returns:
            Dictionary with confidence metrics
        """
        try:
            # Normalize scores to [0, 1] range
            similarity_score = max(0.0, min(1.0, similarity_score))
            reranker_score = max(0.0, min(1.0, reranker_score))
            intent_similarity = max(0.0, min(1.0, intent_similarity))
            entity_match_score = max(0.0, min(1.0, entity_match_score))
            
            # CRITICAL: For queries requiring precise matching, entity score is make-or-break
            if requires_precise_matching:
                if entity_match_score < 0.8:
                    # Force low confidence for poor entity matches on critical queries
                    adjusted_score = 0.0
                    confidence_level = "low"
                    decision = "llm_fallback"
                    
                    logger.info(f"🚨 ENTITY MISMATCH DETECTED: entity_score={entity_match_score:.3f} < 0.8, forcing LLM")
                    
                    return {
                        "confidence_score": 0.0,
                        "confidence_level": confidence_level,
                        "cache_decision": decision,
                        "component_scores": {
                            "similarity": float(similarity_score),
                            "reranker": float(reranker_score),
                            "intent": float(intent_similarity),
                            "entity": float(entity_match_score)
                        },
                        "weights_used": self.weights.copy(),
                        "raw_aggregated": 0.0,
                        "entity_override": True
                    }
            
            # Compute weighted aggregated score
            aggregated_score = (
                self.weights["similarity"] * similarity_score +
                self.weights["reranker"] * reranker_score +
                self.weights["intent"] * intent_similarity +
                self.weights["entity"] * entity_match_score
            )
            
            # Apply confidence boosting/penalty based on score patterns
            adjusted_score = self._apply_confidence_adjustments(
                aggregated_score,
                similarity_score,
                reranker_score,
                intent_similarity,
                entity_match_score,
                requires_precise_matching
            )
            
            # Determine confidence level and decision
            confidence_level = self._get_confidence_level(adjusted_score)
            decision = self._make_cache_decision(adjusted_score)
            
            result = {
                "confidence_score": float(adjusted_score),
                "confidence_level": confidence_level,
                "cache_decision": decision,
                "component_scores": {
                    "similarity": float(similarity_score),
                    "reranker": float(reranker_score),
                    "intent": float(intent_similarity),
                    "entity": float(entity_match_score)
                },
                "weights_used": self.weights.copy(),
                "raw_aggregated": float(aggregated_score),
                "entity_override": False
            }
            
            logger.debug(f"🧮 Confidence computed: {adjusted_score:.3f} ({confidence_level}) -> {decision}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to compute confidence: {e}")
            return {
                "confidence_score": 0.0,
                "confidence_level": "low",
                "cache_decision": "llm_fallback",
                "component_scores": {
                    "similarity": 0.0,
                    "reranker": 0.0,
                    "intent": 0.0,
                    "entity": 0.0
                },
                "weights_used": self.weights.copy(),
                "raw_aggregated": 0.0,
                "entity_override": False
            }
    
    def _apply_confidence_adjustments(self,
                                     base_score: float,
                                     similarity: float,
                                     reranker: float,
                                     intent: float,
                                     entity: float,
                                     requires_precise_matching: bool = False) -> float:
        """
        Apply confidence adjustments based on score patterns
        """
        adjusted_score = base_score
        
        # CRITICAL: Heavy penalty for poor entity matching on action queries
        if requires_precise_matching and entity < 0.9:
            adjusted_score *= 0.1  # Massive penalty
            logger.debug("📉 Massive penalty: poor entity match on critical query")
        
        # Boost confidence if all scores are consistently high
        if all(score > 0.8 for score in [similarity, reranker, intent, entity]):
            adjusted_score *= 1.1
            logger.debug("📈 Confidence boosted: all scores consistently high")
        
        # Boost confidence if top scores agree strongly
        if similarity > 0.9 and reranker > 0.9:
            adjusted_score *= 1.05
            logger.debug("📈 Confidence boosted: similarity and reranker agree strongly")
        
        # Penalize confidence if scores disagree significantly
        score_variance = np.var([similarity, reranker, intent, entity])
        if score_variance > 0.15:  # High variance indicates disagreement
            adjusted_score *= 0.9
            logger.debug("📉 Confidence penalized: high score variance detected")
        
        # Penalize confidence if intent differs significantly
        if intent < 0.3 and (similarity > 0.7 or reranker > 0.7):
            adjusted_score *= 0.8
            logger.debug("📉 Confidence penalized: intent mismatch detected")
        
        # BOOST confidence if entities match perfectly
        if entity > 0.95:
            adjusted_score *= 1.1
            logger.debug("📈 Confidence boosted: perfect entity match")
        
        # PENALIZE heavily if entities don't match on action queries
        if requires_precise_matching and entity < 0.7:
            adjusted_score *= 0.3
            logger.debug("📉 Heavy penalty: entity mismatch on action query")
        
        # Ensure score stays in [0, 1] range
        return max(0.0, min(1.0, adjusted_score))
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level category"""
        if score >= self.thresholds["high_confidence"]:
            return "high"
        elif score >= self.thresholds["medium_confidence"]:
            return "medium"
        else:
            return "low"
    
    def _make_cache_decision(self, score: float) -> str:
        """Make cache decision based on confidence score"""
        if score >= self.thresholds["high_confidence"]:
            return "cache_hit"
        elif score >= self.thresholds["medium_confidence"]:
            return "optional_llm"  # Could use cache or call LLM based on policy
        else:
            return "llm_fallback"
    
    def update_thresholds(self, 
                         high_confidence: float = None,
                         medium_confidence: float = None):
        """
        Update confidence thresholds
        
        Args:
            high_confidence: Threshold for high confidence cache hits
            medium_confidence: Threshold for medium confidence decisions
        """
        if high_confidence is not None:
            self.thresholds["high_confidence"] = max(0.0, min(1.0, high_confidence))
        
        if medium_confidence is not None:
            self.thresholds["medium_confidence"] = max(0.0, min(1.0, medium_confidence))
        
        logger.info(f"🎛️ Updated confidence thresholds: {self.thresholds}")
    
    def update_weights(self, 
                      similarity: float = None,
                      reranker: float = None,
                      intent: float = None,
                      entity: float = None):
        """
        Update component weights (must sum to 1.0)
        
        Args:
            similarity: Weight for cosine similarity
            reranker: Weight for cross-encoder reranker
            intent: Weight for intent classification
            entity: Weight for entity matching
        """
        new_weights = self.weights.copy()
        
        if similarity is not None:
            new_weights["similarity"] = max(0.0, min(1.0, similarity))
        if reranker is not None:
            new_weights["reranker"] = max(0.0, min(1.0, reranker))
        if intent is not None:
            new_weights["intent"] = max(0.0, min(1.0, intent))
        if entity is not None:
            new_weights["entity"] = max(0.0, min(1.0, entity))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in new_weights.items()}
            logger.info(f"⚖️ Updated component weights: {self.weights}")
        else:
            logger.warning("⚠️ Invalid weights provided, keeping current weights")
    
    def explain_decision(self, confidence_result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of confidence decision
        
        Args:
            confidence_result: Result from compute_confidence()
            
        Returns:
            Explanation string
        """
        score = confidence_result["confidence_score"]
        level = confidence_result["confidence_level"]
        decision = confidence_result["cache_decision"]
        components = confidence_result["component_scores"]
        entity_override = confidence_result.get("entity_override", False)
        
        explanation = f"Confidence: {score:.3f} ({level})\n"
        explanation += f"Decision: {decision}\n"
        
        if entity_override:
            explanation += "⚠️ ENTITY OVERRIDE: Poor entity match forced LLM fallback\n"
        
        explanation += "\nComponent Scores:\n"
        
        for component, component_score in components.items():
            weight = self.weights[component]
            contribution = weight * component_score
            explanation += f"  {component.capitalize()}: {component_score:.3f} "
            explanation += f"(weight: {weight:.2f}, contribution: {contribution:.3f})\n"
        
        return explanation