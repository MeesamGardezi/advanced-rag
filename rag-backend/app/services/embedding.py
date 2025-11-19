"""
Embedding service for generating vector representations of text.
Uses sentence-transformers for local embedding generation.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        """Initialize the embedding model."""
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            print(f"Loading embedding model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model)
            print(f"✅ Embedding model loaded successfully (dimension: {settings.vector_dimension})")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Convert to list and ensure it's the right dimension
        embedding_list = embedding.tolist()
        
        if len(embedding_list) != settings.vector_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {settings.vector_dimension}, "
                f"got {len(embedding_list)}"
            )
        
        return embedding_list
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return []
        
        # Batch encoding is more efficient
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 10)
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        # Validate dimensions
        for idx, emb in enumerate(embeddings_list):
            if len(emb) != settings.vector_dimension:
                raise ValueError(
                    f"Embedding {idx} dimension mismatch: expected {settings.vector_dimension}, "
                    f"got {len(emb)}"
                )
        
        return embeddings_list
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.vector_dimension


# Global instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    Ensures only one model is loaded in memory.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service