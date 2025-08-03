"""
FAISS Index Management for Semantic Cache
Handles vector storage, retrieval, and similarity search
"""

import faiss
import numpy as np
import logging
import pickle
import os
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class FAISSIndexManager:
    """FAISS index manager for semantic vector search"""
    
    def __init__(self, embedding_dim: int = 768, index_type: str = "flat"):
        """
        Initialize FAISS index manager
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = {}  # Store query-response pairs and metadata
        self.id_counter = 0
        
    def initialize_index(self):
        """Initialize the FAISS index"""
        try:
            logger.info(f"🔧 Initializing FAISS index (dim={self.embedding_dim}, type={self.index_type})")
            
            if self.index_type == "flat":
                # Flat index for exact search (good for small datasets)
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
            elif self.index_type == "ivf":
                # IVF index for approximate search (good for large datasets)
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                nlist = 100  # Number of clusters
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                self.index.nprobe = 10  # Number of clusters to search
            elif self.index_type == "hnsw":
                # HNSW index for very fast approximate search
                M = 32  # Number of connections
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info("✅ FAISS index initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS index: {e}")
            raise
    
    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Add vectors to the index with metadata
        
        Args:
            embeddings: Array of embeddings to add
            metadata_list: List of metadata dictionaries for each embedding
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        try:
            # Ensure embeddings are normalized for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Train index if needed (for IVF)
            if self.index_type == "ivf" and not self.index.is_trained:
                if embeddings.shape[0] >= 100:  # Need enough samples to train
                    self.index.train(embeddings)
                    logger.info("🎯 FAISS IVF index trained")
            
            # Add vectors to index
            start_id = self.id_counter
            self.index.add(embeddings)
            
            # Store metadata
            for i, metadata in enumerate(metadata_list):
                self.metadata[start_id + i] = metadata
            
            self.id_counter += len(embeddings)
            
            logger.debug(f"➕ Added {len(embeddings)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"❌ Failed to add vectors to index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar vectors in the index
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        if self.index.ntotal == 0:
            logger.warning("⚠️ Index is empty, no results to return")
            return []
        
        try:
            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search the index
            similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1:  # Valid index
                    metadata = self.metadata.get(idx, {})
                    results.append((int(idx), float(similarity), metadata))
            
            logger.debug(f"🔍 Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search index: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metadata_count": len(self.metadata)
        }
    
    def save_index(self, filepath: str):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_counter': self.id_counter,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type
                }, f)
            
            logger.info(f"💾 Saved FAISS index to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
            raise
    
    def load_index(self, filepath: str):
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            if os.path.exists(f"{filepath}.faiss"):
                self.index = faiss.read_index(f"{filepath}.faiss")
                
                # Load metadata
                if os.path.exists(f"{filepath}.metadata"):
                    with open(f"{filepath}.metadata", 'rb') as f:
                        data = pickle.load(f)
                        self.metadata = data['metadata']
                        self.id_counter = data['id_counter']
                        self.embedding_dim = data['embedding_dim']
                        self.index_type = data['index_type']
                
                logger.info(f"📂 Loaded FAISS index from {filepath}")
                return True
            else:
                logger.info(f"📂 Index file not found: {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def clear_index(self):
        """Clear the index and metadata"""
        try:
            self.initialize_index()
            self.metadata.clear()
            self.id_counter = 0
            logger.info("🗑️ FAISS index cleared successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear index: {e}")
            raise