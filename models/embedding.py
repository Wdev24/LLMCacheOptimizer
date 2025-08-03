"""
BGE Embedding Model Loader and Interface
Using BAAI/bge-base-en-v1.5 for semantic embeddings
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class BGEEmbeddingModel:
    """BGE Embedding Model for semantic similarity"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize BGE embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 768  # BGE-base embedding dimension
        
    def load_model(self):
        """Load the BGE embedding model"""
        try:
            logger.info(f"📥 Loading BGE embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ BGE embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load BGE embedding model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for input texts
        
        Args:
            texts: Single text or list of texts to embed
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            
        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Failed to compute similarity: {e}")
            return 0.0
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        
        return text