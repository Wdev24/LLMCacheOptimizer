"""
Cross-Encoder Reranker Model
Using cross-encoder/ms-marco-MiniLM-L-6-v2 for semantic reranking
"""

from sentence_transformers import CrossEncoder
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-Encoder model for reranking retrieved results"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize Cross-Encoder reranker
        
        Args:
            model_name: HuggingFace cross-encoder model identifier
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load the cross-encoder model"""
        try:
            logger.info(f"📥 Loading Cross-Encoder reranker: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("✅ Cross-Encoder reranker loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load cross-encoder model: {e}")
            raise
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rerank candidate responses based on semantic relevance to query
        
        Args:
            query: Input query
            candidates: List of candidate responses
            top_k: Number of top candidates to return
            
        Returns:
            List of (candidate, score) tuples sorted by relevance
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not candidates:
            return []
        
        try:
            # Prepare query-candidate pairs
            pairs = [[query, candidate] for candidate in candidates]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Convert scores to probabilities (sigmoid)
            scores = 1 / (1 + np.exp(-scores))
            
            # Create candidate-score pairs
            candidate_scores = list(zip(candidates, scores))
            
            # Sort by score (descending) and take top_k
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_results = candidate_scores[:top_k]
            
            logger.debug(f"🔄 Reranked {len(candidates)} candidates, returning top {len(reranked_results)}")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"❌ Failed to rerank candidates: {e}")
            # Return original candidates with zero scores as fallback
            return [(candidate, 0.0) for candidate in candidates[:top_k]]
    
    def compute_relevance_score(self, query: str, candidate: str) -> float:
        """
        Compute relevance score between query and single candidate
        
        Args:
            query: Input query
            candidate: Candidate response
            
        Returns:
            Relevance score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Get relevance score
            score = self.model.predict([[query, candidate]])[0]
            
            # Convert to probability
            score = 1 / (1 + np.exp(-score))
            
            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, float(score)))
            
            return score
            
        except Exception as e:
            logger.error(f"❌ Failed to compute relevance score: {e}")
            return 0.0
    
    def batch_score(self, query: str, candidates: List[str]) -> List[float]:
        """
        Compute relevance scores for multiple candidates efficiently
        
        Args:
            query: Input query
            candidates: List of candidate responses
            
        Returns:
            List of relevance scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not candidates:
            return []
        
        try:
            # Prepare query-candidate pairs
            pairs = [[query, candidate] for candidate in candidates]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Convert scores to probabilities
            scores = 1 / (1 + np.exp(-scores))
            
            # Ensure scores are in [0, 1] range
            scores = [max(0.0, min(1.0, float(score))) for score in scores]
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Failed to compute batch scores: {e}")
            return [0.0] * len(candidates)