"""Base reranker interface."""

from abc import ABC, abstractmethod
from typing import List

from meulex.core.vector.base import Document


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents with updated scores
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the reranker and cleanup resources."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the reranker name."""
        pass
