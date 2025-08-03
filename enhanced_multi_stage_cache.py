"""
Enhanced Multi-Stage Semantic Cache
Implements selected advanced features from the enterprise architecture
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import time
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class QueryDecomposition:
    """Decomposed query components"""
    original_query: str
    atomic_intents: List[str]
    query_type: str
    complexity_score: float
    alternate_phrasings: List[str]

@dataclass
class RetrievalCandidate:
    """Enhanced candidate with multiple scores"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    reranker_score: float
    graph_centrality: float
    confidence_raw: float
    confidence_calibrated: float

class QueryDecomposer:
    """Stage 0: Query Understanding & Context Expansion"""
    
    def __init__(self):
        self.compound_patterns = [
            r'\band\b', r'\bplus\b', r'\balso\b', r'\bthen\b',
            r'\bafter\b', r'\bbefore\b', r'\bwhile\b'
        ]
        
        self.question_types = {
            'what': 'definition',
            'how': 'explanation', 
            'why': 'reasoning',
            'when': 'temporal',
            'where': 'location',
            'who': 'entity'
        }
    
    def decompose_query(self, query: str) -> QueryDecomposition:
        """Decompose query into atomic components"""
        import re
        
        # Detect compound queries
        is_compound = any(re.search(pattern, query.lower()) for pattern in self.compound_patterns)
        
        if is_compound:
            # Simple decomposition (can be enhanced with T5 later)
            atomic_intents = re.split(r'\s+(?:and|plus|also|then)\s+', query.lower())
        else:
            atomic_intents = [query]
        
        # Determine query type
        first_word = query.lower().split()[0] if query.split() else ""
        query_type = self.question_types.get(first_word, "statement")
        
        # Calculate complexity score
        complexity_score = min(1.0, len(query.split()) / 20.0 + len(atomic_intents) / 5.0)
        
        # Generate alternate phrasings (simple rule-based)
        alternate_phrasings = self._generate_alternates(query)
        
        return QueryDecomposition(
            original_query=query,
            atomic_intents=atomic_intents,
            query_type=query_type,
            complexity_score=complexity_score,
            alternate_phrasings=alternate_phrasings
        )
    
    def _generate_alternates(self, query: str) -> List[str]:
        """Generate alternate phrasings"""
        alternates = []
        query_lower = query.lower()
        
        # Simple transformations
        transformations = [
            ("what is", "define"),
            ("how does", "explain how"),
            ("tell me about", "what is"),
            ("can you explain", "explain"),
            ("i want to know", "what is")
        ]
        
        for old, new in transformations:
            if old in query_lower:
                alternate = query_lower.replace(old, new)
                alternates.append(alternate.capitalize())
        
        return alternates[:3]  # Limit to top 3

class MultiVectorRetriever:
    """Stage 1: Enhanced Hybrid Retrieval"""
    
    def __init__(self, embedding_model, faiss_manager):
        self.embedding_model = embedding_model
        self.faiss_manager = faiss_manager
        self.context_weight = 0.3
        self.semantic_weight = 0.7
    
    def retrieve_candidates(self, 
                          query_decomp: QueryDecomposition, 
                          top_k: int = 20) -> List[RetrievalCandidate]:
        """Enhanced retrieval with multiple strategies"""
        
        all_candidates = []
        
        # Strategy 1: Main query
        main_candidates = self._retrieve_for_query(query_decomp.original_query, top_k)
        all_candidates.extend(main_candidates)
        
        # Strategy 2: Atomic intents (for compound queries)
        for intent in query_decomp.atomic_intents[:2]:  # Limit to avoid explosion
            if intent != query_decomp.original_query.lower():
                intent_candidates = self._retrieve_for_query(intent, top_k // 2)
                all_candidates.extend(intent_candidates)
        
        # Strategy 3: Alternate phrasings
        for alt in query_decomp.alternate_phrasings[:2]:
            alt_candidates = self._retrieve_for_query(alt, top_k // 3)
            all_candidates.extend(alt_candidates)
        
        # Deduplicate and merge
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        return unique_candidates[:top_k]
    
    def _retrieve_for_query(self, query: str, k: int) -> List[RetrievalCandidate]:
        """Retrieve candidates for a single query"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(query)
            
            # Search FAISS
            search_results = self.faiss_manager.search(embedding, top_k=k)
            
            candidates = []
            for idx, similarity, metadata in search_results:
                candidate = RetrievalCandidate(
                    content=metadata.get("response", ""),
                    metadata=metadata,
                    similarity_score=similarity,
                    reranker_score=0.0,  # Will be filled by reranker
                    graph_centrality=0.0,  # Will be filled by graph module
                    confidence_raw=0.0,
                    confidence_calibrated=0.0
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed for query '{query}': {e}")
            return []
    
    def _deduplicate_candidates(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Remove duplicate candidates"""
        seen_content = set()
        unique_candidates = []
        
        for candidate in candidates:
            content_hash = hash(candidate.content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_candidates.append(candidate)
        
        return unique_candidates

class EnsembleReranker:
    """Stage 2: Enhanced Cross-Encoder Reranking"""
    
    def __init__(self, reranker_model):
        self.reranker_model = reranker_model
        self.graph_weight = 0.2
        self.reranker_weight = 0.8
    
    def rerank_candidates(self, 
                         query: str, 
                         candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Enhanced reranking with graph centrality"""
        
        if not candidates:
            return candidates
        
        # Step 1: Cross-encoder reranking
        contents = [c.content for c in candidates]
        reranker_results = self.reranker_model.rerank(query, contents, top_k=len(contents))
        
        # Update reranker scores
        reranker_scores = {content: score for content, score in reranker_results}
        for candidate in candidates:
            candidate.reranker_score = reranker_scores.get(candidate.content, 0.0)
        
        # Step 2: Graph centrality computation
        centrality_scores = self._compute_graph_centrality(candidates)
        for i, candidate in enumerate(candidates):
            candidate.graph_centrality = centrality_scores[i]
        
        # Step 3: Combined scoring
        for candidate in enumerate(candidates):
            combined_score = (
                self.reranker_weight * candidate.reranker_score +
                self.graph_weight * candidate.graph_centrality
            )
            candidate.confidence_raw = combined_score
        
        # Sort by combined score
        candidates.sort(key=lambda x: x.confidence_raw, reverse=True)
        
        return candidates
    
    def _compute_graph_centrality(self, candidates: List[RetrievalCandidate]) -> List[float]:
        """Compute graph centrality scores"""
        n = len(candidates)
        if n <= 1:
            return [1.0] * n
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Simple similarity based on shared words
                words_i = set(candidates[i].content.lower().split())
                words_j = set(candidates[j].content.lower().split())
                
                if words_i and words_j:
                    jaccard_sim = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                    similarity_matrix[i][j] = jaccard_sim
                    similarity_matrix[j][i] = jaccard_sim
        
        # Simple PageRank-like centrality
        centrality = np.ones(n)
        for _ in range(3):  # 3 iterations
            new_centrality = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        new_centrality[i] += similarity_matrix[i][j] * centrality[j]
            centrality = new_centrality / (np.sum(new_centrality) + 1e-8)
        
        return centrality.tolist()

class BayesianCalibrator:
    """Stage 3: Bayesian Confidence Calibration"""
    
    def __init__(self):
        self.temperature = 2.0  # Temperature scaling parameter
        self.calibration_history = defaultdict(list)
    
    def calibrate_confidence(self, 
                           candidates: List[RetrievalCandidate],
                           query_complexity: float) -> List[RetrievalCandidate]:
        """Apply Bayesian calibration to confidence scores"""
        
        for candidate in candidates:
            # Temperature scaling
            calibrated_score = self._temperature_scaling(candidate.confidence_raw)
            
            # Complexity adjustment
            complexity_penalty = min(0.2, query_complexity * 0.1)
            calibrated_score *= (1.0 - complexity_penalty)
            
            # Update calibrated confidence
            candidate.confidence_calibrated = max(0.0, min(1.0, calibrated_score))
        
        return candidates
    
    def _temperature_scaling(self, raw_score: float) -> float:
        """Apply temperature scaling to raw confidence"""
        if raw_score <= 0:
            return 0.0
        
        # Convert to logit, apply temperature, convert back
        logit = np.log(raw_score / (1.0 - raw_score + 1e-8))
        calibrated_logit = logit / self.temperature
        calibrated_score = 1.0 / (1.0 + np.exp(-calibrated_logit))
        
        return calibrated_score

class ContextAwareLLMFallback:
    """Stage 6: Enhanced LLM Fallback with Context"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_cache = {}
        self.cost_budget = 100  # Daily API call budget
        self.calls_today = 0
    
    def should_fallback_to_llm(self, 
                              best_confidence: float, 
                              query_complexity: float,
                              hour_of_day: int) -> bool:
        """Adaptive thresholding for LLM fallback"""
        
        # Base threshold
        base_threshold = 0.6
        
        # Adjust based on complexity
        complexity_adjustment = query_complexity * 0.1
        
        # Adjust based on time of day (stricter during peak hours)
        time_adjustment = 0.1 if 9 <= hour_of_day <= 17 else 0.0
        
        # Adjust based on budget
        budget_adjustment = 0.2 if self.calls_today > self.cost_budget * 0.8 else 0.0
        
        dynamic_threshold = base_threshold + complexity_adjustment + time_adjustment + budget_adjustment
        
        return best_confidence < dynamic_threshold
    
    def generate_with_context(self, 
                            query: str, 
                            top_candidates: List[RetrievalCandidate]) -> str:
        """Generate LLM response with retrieved context"""
        
        # Check prompt cache first
        cache_key = hash(query + str([c.content[:50] for c in top_candidates[:3]]))
        if cache_key in self.prompt_cache:
            logger.info("🎯 Using cached LLM prompt")
            return self.prompt_cache[cache_key]
        
        # Build context-aware prompt
        context = "\n".join([
            f"- {candidate.content[:200]}..." 
            for candidate in top_candidates[:3]
        ])
        
        enhanced_prompt = f"""
Based on this related information:
{context}

Please answer the following query: {query}

If the above information helps answer the query, use it as context. 
If not, provide a direct answer to the query.
"""
        
        try:
            response = self.llm_client.generate_response(enhanced_prompt)
            self.calls_today += 1
            
            # Cache the response
            self.prompt_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Enhanced LLM fallback failed: {e}")
            return "I apologize, but I'm unable to process your request at the moment."

class EnhancedSemanticCacheEngine:
    """Main orchestrator for the enhanced multi-stage cache"""
    
    def __init__(self, 
                 embedding_model, 
                 reranker_model, 
                 faiss_manager, 
                 llm_client,
                 entity_extractor=None):
        
        # Initialize all stages
        self.query_decomposer = QueryDecomposer()
        self.multi_retriever = MultiVectorRetriever(embedding_model, faiss_manager)
        self.ensemble_reranker = EnsembleReranker(reranker_model)
        self.bayesian_calibrator = BayesianCalibrator()
        self.enhanced_llm = ContextAwareLLMFallback(llm_client)
        self.entity_extractor = entity_extractor
        
        # Statistics
        self.stage_timings = defaultdict(list)
        
    def process_query_enhanced(self, query: str) -> Dict[str, Any]:
        """Process query through enhanced multi-stage pipeline"""
        
        start_time = time.time()
        stage_times = {}
        
        try:
            # Stage 0: Query Understanding
            stage_start = time.time()
            query_decomp = self.query_decomposer.decompose_query(query)
            stage_times['decomposition'] = time.time() - stage_start
            
            # Stage 1: Multi-Vector Retrieval
            stage_start = time.time()
            candidates = self.multi_retriever.retrieve_candidates(query_decomp, top_k=15)
            stage_times['retrieval'] = time.time() - stage_start
            
            if not candidates:
                return self._fallback_response(query, stage_times, start_time)
            
            # Stage 2: Ensemble Reranking
            stage_start = time.time()
            reranked_candidates = self.ensemble_reranker.rerank_candidates(query, candidates)
            stage_times['reranking'] = time.time() - stage_start
            
            # Stage 3: Bayesian Calibration
            stage_start = time.time()
            calibrated_candidates = self.bayesian_calibrator.calibrate_confidence(
                reranked_candidates, query_decomp.complexity_score
            )
            stage_times['calibration'] = time.time() - stage_start
            
            # Decision making
            best_candidate = calibrated_candidates[0]
            best_confidence = best_candidate.confidence_calibrated
            
            # Enhanced entity validation (if available)
            if self.entity_extractor:
                entity_validation = self._validate_entities(query, best_candidate)
                if not entity_validation['valid']:
                    logger.info(f"🚨 Entity validation failed: {entity_validation['reason']}")
                    return self._fallback_response(query, stage_times, start_time)
            
            # Adaptive LLM fallback decision
            current_hour = int(time.time() / 3600) % 24
            should_fallback = self.enhanced_llm.should_fallback_to_llm(
                best_confidence, query_decomp.complexity_score, current_hour
            )
            
            if should_fallback:
                stage_start = time.time()
                llm_response = self.enhanced_llm.generate_with_context(query, calibrated_candidates[:3])
                stage_times['llm_generation'] = time.time() - stage_start
                
                return {
                    'response': llm_response,
                    'source': 'enhanced_llm',
                    'confidence': 0.0,
                    'query_decomposition': query_decomp.__dict__,
                    'stage_timings': stage_times,
                    'total_time': time.time() - start_time
                }
            else:
                return {
                    'response': best_candidate.content,
                    'source': 'enhanced_cache',
                    'confidence': best_confidence,
                    'query_decomposition': query_decomp.__dict__,
                    'candidate_metadata': best_candidate.metadata,
                    'stage_timings': stage_times,
                    'total_time': time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"❌ Enhanced pipeline failed: {e}")
            return self._fallback_response(query, stage_times, start_time)
    
    def _validate_entities(self, query: str, candidate: RetrievalCandidate) -> Dict[str, Any]:
        """Enhanced entity validation"""
        if not self.entity_extractor:
            return {'valid': True, 'reason': 'no_validator'}
        
        try:
            query_entities = self.entity_extractor.extract_entities(query)
            cached_query = candidate.metadata.get('query', '')
            cached_entities = self.entity_extractor.extract_entities(cached_query)
            
            entity_similarity = self.entity_extractor.calculate_entity_similarity(
                query_entities, cached_entities
            )
            
            requires_precise = self.entity_extractor.requires_precise_matching(query, cached_query)
            
            if requires_precise and entity_similarity < 0.8:
                return {
                    'valid': False, 
                    'reason': f'Entity mismatch: {entity_similarity:.3f} < 0.8',
                    'query_entities': query_entities,
                    'cached_entities': cached_entities
                }
            
            return {'valid': True, 'reason': 'entities_match'}
            
        except Exception as e:
            logger.error(f"❌ Entity validation failed: {e}")
            return {'valid': True, 'reason': 'validation_error'}
    
    def _fallback_response(self, query: str, stage_times: Dict, start_time: float) -> Dict[str, Any]:
        """Generate fallback response"""
        try:
            response = self.enhanced_llm.generate_with_context(query, [])
            stage_times['llm_fallback'] = time.time() - start_time
            
            return {
                'response': response,
                'source': 'llm_fallback',
                'confidence': 0.0,
                'stage_timings': stage_times,
                'total_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'response': "I apologize, but I'm unable to process your request.",
                'source': 'error',
                'confidence': 0.0,
                'stage_timings': stage_times,
                'total_time': time.time() - start_time
            }