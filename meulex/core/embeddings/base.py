"""Base embeddings interface."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract base class for embeddings providers."""
    
    @abstractmethod
    async def embed_async_single(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the embedder and cleanup resources."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
