"""Base vector store interface and document model."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for vector storage."""
    
    id: str = Field(description="Unique document identifier")
    content: str = Field(description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Document embedding")
    score: Optional[float] = Field(default=None, description="Similarity score")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add embeddings to the vector store.
        
        Args:
            texts: Texts to store
            embeddings: Corresponding embeddings
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        pass
    
    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Perform similarity search by vector.
        
        Args:
            embedding: Query embedding
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: List[str],
        **kwargs: Any
    ) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: Document IDs to delete
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information.
        
        Returns:
            Collection information
        """
        pass
    
    @abstractmethod
    async def create_collection(self, **kwargs: Any) -> bool:
        """Create collection if it doesn't exist.
        
        Args:
            **kwargs: Collection parameters
            
        Returns:
            True if successful
        """
        pass
