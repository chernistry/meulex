"""Dense retrieval using vector similarity search."""

import logging
from typing import Dict, List, Optional

from meulex.config.settings import Settings
from meulex.core.embeddings.base import BaseEmbedder
from meulex.core.vector.base import Document, VectorStore
from meulex.observability import DOCUMENTS_RETRIEVED, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class DenseRetriever:
    """Dense retriever using vector similarity search."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        settings: Settings
    ) -> None:
        """Initialize dense retriever.
        
        Args:
            embedder: Embeddings provider
            vector_store: Vector store
            settings: Application settings
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings
        
        logger.info(
            "Dense retriever initialized",
            extra={
                "embedder": embedder.model_name,
                "vector_store": type(vector_store).__name__
            }
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """Retrieve documents using dense vector search.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of retrieved documents
        """
        with tracer.start_as_current_span("dense_retrieve") as span:
            top_k = top_k or self.settings.default_top_k
            
            span.set_attribute("query_length", len(query))
            span.set_attribute("top_k", top_k)
            span.set_attribute("has_filter", filter is not None)
            
            try:
                # Generate query embedding
                query_embedding = await self.embedder.embed_async_single(query)
                
                # Search vector store
                documents = await self.vector_store.similarity_search_by_vector(
                    embedding=query_embedding,
                    k=top_k,
                    filter=filter
                )
                
                # Record metrics
                DOCUMENTS_RETRIEVED.labels(strategy="dense").inc(len(documents))
                
                logger.info(
                    f"Retrieved {len(documents)} documents",
                    extra={
                        "query_length": len(query),
                        "top_k": top_k,
                        "results_count": len(documents)
                    }
                )
                
                span.set_attribute("results_count", len(documents))
                span.set_attribute("success", True)
                
                return documents
                
            except Exception as e:
                logger.error(f"Dense retrieval failed: {e}")
                span.set_attribute("error", str(e))
                raise
    
    async def close(self) -> None:
        """Close the retriever and cleanup resources."""
        try:
            await self.embedder.close()
            await self.vector_store.close()
            logger.info("Dense retriever closed")
        except Exception as e:
            logger.warning(f"Error closing dense retriever: {e}")
