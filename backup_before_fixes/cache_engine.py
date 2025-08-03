"""
Main Semantic Cache Engine
Orchestrates all components for intelligent cache decisions
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
import time

from models.embedding import BGEEmbeddingModel
from models.reranker import CrossEncoderReranker
from models.classifier import IntentClassifier
from core.faiss_index import FAISSIndexManager
from core.confidence_logic import ConfidenceAggregator
from llm.together_api import TogetherAIClient

logger = logging.getLogger(__name__)

class SemanticCacheEngine:
    """Main semantic cache engine coordinating all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic cache engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.embedding_model = BGEEmbeddingModel()
        self.reranker = CrossEncoderReranker()
        self.intent_classifier = IntentClassifier()
        self.faiss_manager = FAISSIndexManager(embedding_dim=768)
        self.confidence_aggregator = ConfidenceAggregator()
        self.llm_client = TogetherAIClient()
        
        # Cache statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "average_confidence": 0.0,
            "average_response_time": 0.0
        }
        
        # Cache storage path
        self.cache_path = self.config.get("cache_path", "data/semantic_cache")
        
    def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Semantic Cache Engine...")
            
            # Load models
            logger.info("📥 Loading models...")
            self.embedding_model.load_model()
            self.reranker.load_model()
            self.intent_classifier.load_model()
            
            # Initialize FAISS index
            self.faiss_manager.initialize_index()
            
            # Try to load existing cache
            if os.path.exists(f"{self.cache_path}.faiss"):
                logger.info("📂 Loading existing cache...")
                self.faiss_manager.load_index(self.cache_path)
            else:
                logger.info("🆕 Creating new cache...")
                self._load_example_data()
            
            logger.info("✅ Semantic Cache Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache engine: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query through semantic cache system
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with response, source, confidence, and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing query: '{query}'")
            
            # Update statistics
            self.stats["total_queries"] += 1
            
            # Preprocess query
            processed_query = self.embedding_model.preprocess_text(query)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(processed_query)
            
            # Search for similar cached responses
            search_results = self.faiss_manager.search(query_embedding, top_k=10)
            
            if not search_results:
                logger.info("📭 No cached results found, using LLM fallback")
                return self._handle_llm_fallback(query, start_time)
            
            # Extract candidates for reranking
            candidates = []
            similarities = []
            metadata_list = []
            
            for idx, similarity, metadata in search_results:
                candidates.append(metadata.get("response", ""))
                similarities.append(similarity)
                metadata_list.append(metadata)
            
            # Rerank candidates
            reranked_results = self.reranker.rerank(query, candidates, top_k=5)
            
            if not reranked_results:
                logger.info("📭 No valid reranked results, using LLM fallback")
                return self._handle_llm_fallback(query, start_time)
            
            # Get best candidate
            best_candidate, reranker_score = reranked_results[0]
            best_similarity = similarities[0]  # Top similarity score
            
            # Find corresponding metadata
            best_metadata = None
            for metadata in metadata_list:
                if metadata.get("response") == best_candidate:
                    best_metadata = metadata
                    break
            
            if not best_metadata:
                logger.warning("⚠️ Could not find metadata for best candidate")
                return self._handle_llm_fallback(query, start_time)
            
            # Compute intent similarity
            cached_query = best_metadata.get("query", "")
            intent_similarity = self.intent_classifier.compute_intent_similarity(query, cached_query)
            
            # Compute entity match score
            entity_match_score = self._compute_entity_match(query, cached_query)
            
            # Aggregate confidence
            confidence_result = self.confidence_aggregator.compute_confidence(
                similarity_score=best_similarity,
                reranker_score=reranker_score,
                intent_similarity=intent_similarity,
                entity_match_score=entity_match_score
            )
            
            # Make decision based on confidence
            decision = confidence_result["cache_decision"]
            
            if decision == "cache_hit":
                logger.info(f"✅ Cache hit! Confidence: {confidence_result['confidence_score']:.3f}")
                result = self._handle_cache_hit(
                    query, best_candidate, confidence_result, best_metadata, start_time
                )
            elif decision == "optional_llm":
                # For now, use cache for medium confidence
                # In production, this could be configurable
                logger.info(f"🤔 Medium confidence, using cache. Confidence: {confidence_result['confidence_score']:.3f}")
                result = self._handle_cache_hit(
                    query, best_candidate, confidence_result, best_metadata, start_time
                )
            else:
                logger.info(f"🔁 Low confidence, using LLM fallback. Confidence: {confidence_result['confidence_score']:.3f}")
                result = self._handle_llm_fallback(query, start_time)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_queries"] - 1) + processing_time) 
                / self.stats["total_queries"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return self._handle_llm_fallback(query, start_time)
    
    def _handle_cache_hit(self, 
                         query: str, 
                         response: str, 
                         confidence_result: Dict[str, Any],
                         metadata: Dict[str, Any],
                         start_time: float) -> Dict[str, Any]:
        """Handle cache hit scenario"""
        self.stats["cache_hits"] += 1
        
        confidence_score = confidence_result["confidence_score"]
        self.stats["average_confidence"] = (
            (self.stats["average_confidence"] * (self.stats["cache_hits"] - 1) + confidence_score) 
            / self.stats["cache_hits"]
        )
        
        return {
            "response": response,
            "source": "cache",
            "confidence": confidence_score,
            "processing_time": time.time() - start_time,
            "similarity_score": confidence_result["component_scores"]["similarity"],
            "reranker_score": confidence_result["component_scores"]["reranker"],
            "intent_score": confidence_result["component_scores"]["intent"],
            "cached_query": metadata.get("query", ""),
            "top_matches": [{"query": metadata.get("query", ""), "score": confidence_score}]
        }
    
    def _handle_llm_fallback(self, query: str, start_time: float) -> Dict[str, Any]:
        """Handle LLM fallback scenario"""
        try:
            self.stats["llm_calls"] += 1
            
            # Call LLM
            llm_response = self.llm_client.generate_response(query)
            
            # Add to cache for future use
            self._add_to_cache(query, llm_response)
            
            return {
                "response": llm_response,
                "source": "llm",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "similarity_score": 0.0,
                "reranker_score": 0.0,
                "intent_score": 0.0,
                "cached_query": "",
                "top_matches": []
            }
            
        except Exception as e:
            logger.error(f"❌ LLM fallback failed: {e}")
            return {
                "response": "I apologize, but I'm unable to process your request at the moment. Please try again later.",
                "source": "error",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "similarity_score": 0.0,
                "reranker_score": 0.0,
                "intent_score": 0.0,
                "cached_query": "",
                "top_matches": []
            }
    
    def _add_to_cache(self, query: str, response: str):
        """Add query-response pair to cache"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(query)
            
            # Prepare metadata
            metadata = {
                "query": query,
                "response": response,
                "timestamp": time.time(),
                "intent": self.intent_classifier.get_primary_intent(query),
                "entities": self.intent_classifier.extract_entities(query)
            }
            
            # Add to FAISS index
            self.faiss_manager.add_vectors(embedding.reshape(1, -1), [metadata])
            
            logger.debug(f"💾 Added to cache: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"❌ Failed to add to cache: {e}")
    
"""
Main Semantic Cache Engine
Orchestrates all components for intelligent cache decisions
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
import time

from models.embedding import BGEEmbeddingModel
from models.reranker import CrossEncoderReranker
from models.classifier import IntentClassifier
from core.faiss_index import FAISSIndexManager
from core.confidence_logic import ConfidenceAggregator
from llm.together_api import TogetherAIClient

logger = logging.getLogger(__name__)

class SemanticCacheEngine:
    """Main semantic cache engine coordinating all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic cache engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.embedding_model = BGEEmbeddingModel()
        self.reranker = CrossEncoderReranker()
        self.intent_classifier = IntentClassifier()
        self.faiss_manager = FAISSIndexManager(embedding_dim=768)
        self.confidence_aggregator = ConfidenceAggregator()
        self.llm_client = TogetherAIClient()
        
        # Initialize enhanced entity extractor
        try:
            from enhanced_entity_extraction import EnhancedEntityExtractor
            self.entity_extractor = EnhancedEntityExtractor()
            logger.info("✅ Enhanced entity extractor loaded")
        except ImportError:
            logger.warning("⚠️ Enhanced entity extractor not found, using basic extraction")
            self.entity_extractor = None
        
        # Cache statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "entity_overrides": 0,  # Track entity-based rejections
            "false_positive_blocks": 0,  # Track prevented false positives
            "average_confidence": 0.0,
            "average_response_time": 0.0
        }
        
        # Cache storage path
        self.cache_path = self.config.get("cache_path", "data/semantic_cache")
        
    def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Semantic Cache Engine...")
            
            # Load models
            logger.info("📥 Loading models...")
            self.embedding_model.load_model()
            self.reranker.load_model()
            self.intent_classifier.load_model()
            
            # Initialize FAISS index
            self.faiss_manager.initialize_index()
            
            # Try to load existing cache
            if os.path.exists(f"{self.cache_path}.faiss"):
                logger.info("📂 Loading existing cache...")
                self.faiss_manager.load_index(self.cache_path)
            else:
                logger.info("🆕 Creating new cache...")
                self._load_example_data()
            
            logger.info("✅ Semantic Cache Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache engine: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query through semantic cache system with enhanced entity checking
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with response, source, confidence, and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing query: '{query}'")
            
            # Update statistics
            self.stats["total_queries"] += 1
            
            # Preprocess query
            processed_query = self.embedding_model.preprocess_text(query)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(processed_query)
            
            # Search for similar cached responses
            search_results = self.faiss_manager.search(query_embedding, top_k=10)
            
            if not search_results:
                logger.info("📭 No cached results found, using LLM fallback")
                return self._handle_llm_fallback(query, start_time)
            
            # Extract candidates for reranking
            candidates = []
            similarities = []
            metadata_list = []
            
            for idx, similarity, metadata in search_results:
                candidates.append(metadata.get("response", ""))
                similarities.append(similarity)
                metadata_list.append(metadata)
            
            # Rerank candidates
            reranked_results = self.reranker.rerank(query, candidates, top_k=5)
            
            if not reranked_results:
                logger.info("📭 No valid reranked results, using LLM fallback")
                return self._handle_llm_fallback(query, start_time)
            
            # Get best candidate
            best_candidate, reranker_score = reranked_results[0]
            best_similarity = similarities[0]  # Top similarity score
            
            # Find corresponding metadata
            best_metadata = None
            for metadata in metadata_list:
                if metadata.get("response") == best_candidate:
                    best_metadata = metadata
                    break
            
            if not best_metadata:
                logger.warning("⚠️ Could not find metadata for best candidate")
                return self._handle_llm_fallback(query, start_time)
            
            # Compute intent similarity
            cached_query = best_metadata.get("query", "")
            intent_similarity = self.intent_classifier.compute_intent_similarity(query, cached_query)
            
            # 🔥 ENHANCED ENTITY MATCHING
            entity_match_score = self._compute_enhanced_entity_match(query, cached_query)
            
            # Check if this query requires precise entity matching
            requires_precise_matching = False
            if self.entity_extractor:
                requires_precise_matching = self.entity_extractor.requires_precise_matching(query, cached_query)
            
            # Aggregate confidence with enhanced entity checking
            confidence_result = self.confidence_aggregator.compute_confidence(
                similarity_score=best_similarity,
                reranker_score=reranker_score,
                intent_similarity=intent_similarity,
                entity_match_score=entity_match_score,
                requires_precise_matching=requires_precise_matching
            )
            
            # Make decision based on confidence
            decision = confidence_result["cache_decision"]
            entity_override = confidence_result.get("entity_override", False)
            
            if entity_override:
                self.stats["entity_overrides"] += 1
                self.stats["false_positive_blocks"] += 1
                logger.info(f"🚨 ENTITY OVERRIDE: Blocked potential false positive for '{query}'")
                result = self._handle_llm_fallback(query, start_time)
                result["blocked_false_positive"] = True
                result["entity_override_reason"] = f"Entity mismatch: '{query}' vs '{cached_query}'"
                return result
            
            if decision == "cache_hit":
                logger.info(f"✅ Cache hit! Confidence: {confidence_result['confidence_score']:.3f}")
                result = self._handle_cache_hit(
                    query, best_candidate, confidence_result, best_metadata, start_time
                )
            elif decision == "optional_llm":
                # For now, use cache for medium confidence
                # In production, this could be configurable
                logger.info(f"🤔 Medium confidence, using cache. Confidence: {confidence_result['confidence_score']:.3f}")
                result = self._handle_cache_hit(
                    query, best_candidate, confidence_result, best_metadata, start_time
                )
            else:
                logger.info(f"🔁 Low confidence, using LLM fallback. Confidence: {confidence_result['confidence_score']:.3f}")
                result = self._handle_llm_fallback(query, start_time)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_queries"] - 1) + processing_time) 
                / self.stats["total_queries"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return self._handle_llm_fallback(query, start_time)
    
    def _compute_enhanced_entity_match(self, query1: str, query2: str) -> float:
        """Compute enhanced entity matching score"""
        if self.entity_extractor:
            try:
                entities1 = self.entity_extractor.extract_entities(query1)
                entities2 = self.entity_extractor.extract_entities(query2)
                
                entity_similarity = self.entity_extractor.calculate_entity_similarity(entities1, entities2)
                
                logger.debug(f"🔍 Entity analysis:")
                logger.debug(f"   Query1 entities: {entities1}")
                logger.debug(f"   Query2 entities: {entities2}")
                logger.debug(f"   Entity similarity: {entity_similarity:.3f}")
                
                return entity_similarity
                
            except Exception as e:
                logger.error(f"❌ Enhanced entity extraction failed: {e}")
                return self._compute_basic_entity_match(query1, query2)
        else:
            return self._compute_basic_entity_match(query1, query2)
    
    def _compute_basic_entity_match(self, query1: str, query2: str) -> float:
        """Compute basic entity matching score (fallback)"""
        try:
            entities1 = set(self.intent_classifier.extract_entities(query1))
            entities2 = set(self.intent_classifier.extract_entities(query2))
            
            if not entities1 and not entities2:
                return 1.0  # No entities in either, perfect match
            
            if not entities1 or not entities2:
                return 0.0  # One has entities, other doesn't
            
            # Compute Jaccard similarity
            intersection = len(entities1.intersection(entities2))
            union = len(entities1.union(entities2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to compute basic entity match: {e}")
            return 0.0
    
    def _handle_cache_hit(self, 
                         query: str, 
                         response: str, 
                         confidence_result: Dict[str, Any],
                         metadata: Dict[str, Any],
                         start_time: float) -> Dict[str, Any]:
        """Handle cache hit scenario"""
        self.stats["cache_hits"] += 1
        
        confidence_score = confidence_result["confidence_score"]
        self.stats["average_confidence"] = (
            (self.stats["average_confidence"] * (self.stats["cache_hits"] - 1) + confidence_score) 
            / self.stats["cache_hits"]
        )
        
        return {
            "response": response,
            "source": "cache",
            "confidence": confidence_score,
            "processing_time": time.time() - start_time,
            "similarity_score": confidence_result["component_scores"]["similarity"],
            "reranker_score": confidence_result["component_scores"]["reranker"],
            "intent_score": confidence_result["component_scores"]["intent"],
            "entity_score": confidence_result["component_scores"]["entity"],
            "cached_query": metadata.get("query", ""),
            "top_matches": [{"query": metadata.get("query", ""), "score": confidence_score}],
            "confidence_breakdown": confidence_result
        }
    
    def _handle_llm_fallback(self, query: str, start_time: float) -> Dict[str, Any]:
        """Handle LLM fallback scenario"""
        try:
            self.stats["llm_calls"] += 1
            
            # Call LLM
            llm_response = self.llm_client.generate_response(query)
            
            # Add to cache for future use
            self._add_to_cache(query, llm_response)
            
            return {
                "response": llm_response,
                "source": "llm",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "similarity_score": 0.0,
                "reranker_score": 0.0,
                "intent_score": 0.0,
                "entity_score": 0.0,
                "cached_query": "",
                "top_matches": []
            }
            
        except Exception as e:
            logger.error(f"❌ LLM fallback failed: {e}")
            return {
                "response": "I apologize, but I'm unable to process your request at the moment. Please try again later.",
                "source": "error",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "similarity_score": 0.0,
                "reranker_score": 0.0,
                "intent_score": 0.0,
                "entity_score": 0.0,
                "cached_query": "",
                "top_matches": []
            }
    
    def _add_to_cache(self, query: str, response: str):
        """Add query-response pair to cache"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(query)
            
            # Extract entities for metadata
            entities = []
            if self.entity_extractor:
                try:
                    entity_dict = self.entity_extractor.extract_entities(query)
                    entities = [item for sublist in entity_dict.values() for item in sublist]
                except:
                    entities = self.intent_classifier.extract_entities(query)
            else:
                entities = self.intent_classifier.extract_entities(query)
            
            # Prepare metadata
            metadata = {
                "query": query,
                "response": response,
                "timestamp": time.time(),
                "intent": self.intent_classifier.get_primary_intent(query),
                "entities": entities
            }
            
            # Add to FAISS index
            self.faiss_manager.add_vectors(embedding.reshape(1, -1), [metadata])
            
            logger.debug(f"💾 Added to cache: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"❌ Failed to add to cache: {e}")
    
    def _load_example_data(self):
        """Load example cache data"""
        try:
            example_file = "data/example_cache.json"
            if os.path.exists(example_file):
                with open(example_file, 'r') as f:
                    examples = json.load(f)
                
                for example in examples:
                    self._add_to_cache(example["query"], example["response"])
                
                logger.info(f"📚 Loaded {len(examples)} example cache entries")
            else:
                logger.info("📚 No example cache file found, starting with empty cache")
                
        except Exception as e:
            logger.error(f"❌ Failed to load example data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / self.stats["total_queries"]) * 100 if self.stats["total_queries"] > 0 else 0
        
        return {
            **self.stats,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "entity_override_rate": round((self.stats["entity_overrides"] / self.stats["total_queries"]) * 100, 2) if self.stats["total_queries"] > 0 else 0,
            "false_positive_prevention_rate": round((self.stats["false_positive_blocks"] / self.stats["total_queries"]) * 100, 2) if self.stats["total_queries"] > 0 else 0,
            "index_stats": self.faiss_manager.get_stats()
        }
    
    def clear_cache(self):
        """Clear the semantic cache"""
        try:
            self.faiss_manager.clear_index()
            self.stats = {
                "total_queries": 0,
                "cache_hits": 0,
                "llm_calls": 0,
                "entity_overrides": 0,
                "false_positive_blocks": 0,
                "average_confidence": 0.0,
                "average_response_time": 0.0
            }
            logger.info("🗑️ Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            raise
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            self.faiss_manager.save_index(self.cache_path)
            logger.info("💾 Cache saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
            raise
    
    def _load_example_data(self):
        """Load example cache data"""
        try:
            example_file = "data/example_cache.json"
            if os.path.exists(example_file):
                with open(example_file, 'r') as f:
                    examples = json.load(f)
                
                for example in examples:
                    self._add_to_cache(example["query"], example["response"])
                
                logger.info(f"📚 Loaded {len(examples)} example cache entries")
            else:
                logger.info("📚 No example cache file found, starting with empty cache")
                
        except Exception as e:
            logger.error(f"❌ Failed to load example data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / self.stats["total_queries"]) * 100 if self.stats["total_queries"] > 0 else 0
        
        return {
            **self.stats,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "index_stats": self.faiss_manager.get_stats()
        }
    
    def clear_cache(self):
        """Clear the semantic cache"""
        try:
            self.faiss_manager.clear_index()
            self.stats = {
                "total_queries": 0,
                "cache_hits": 0,
                "llm_calls": 0,
                "average_confidence": 0.0,
                "average_response_time": 0.0
            }
            logger.info("🗑️ Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            raise
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            self.faiss_manager.save_index(self.cache_path)
            logger.info("💾 Cache saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
            raise