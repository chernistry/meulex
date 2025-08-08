"""Qdrant vector store implementation."""

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from meulex.config.settings import Settings
from meulex.core.vector.base import Document, VectorStore
from meulex.observability import get_tracer
from meulex.utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class QdrantStore(VectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        settings: Settings,
        collection_name: Optional[str] = None
    ) -> None:
        """Initialize Qdrant store.
        
        Args:
            settings: Application settings
            collection_name: Optional collection name override
        """
        self.settings = settings
        self.collection_name = collection_name or settings.collection_name
        self.vector_size = settings.vector_size
        
        # Initialize client
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.request_timeout_s
        )
        
        logger.info(
            "Qdrant store initialized",
            extra={
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "url": settings.qdrant_url
            }
        )
    
    @retry(
        retry=retry_if_exception_type(ResponseHandlingException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def create_collection(self, **kwargs: Any) -> bool:
        """Create collection if it doesn't exist.
        
        Args:
            **kwargs: Additional collection parameters
            
        Returns:
            True if successful
        """
        with tracer.start_as_current_span("qdrant_create_collection") as span:
            span.set_attribute("collection_name", self.collection_name)
            span.set_attribute("vector_size", self.vector_size)
            
            try:
                # Check if collection exists
                collections = self.client.get_collections()
                existing_names = [col.name for col in collections.collections]
                
                if self.collection_name in existing_names:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return True
                
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfig(
                        default_segment_number=2,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    ),
                    hnsw_config=models.HnswConfig(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                        max_indexing_threads=0,
                        on_disk=False
                    )
                )
                
                logger.info(f"Created collection {self.collection_name}")
                span.set_attribute("created", True)
                return True
                
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                span.set_attribute("error", str(e))
                raise VectorStoreError(
                    f"Failed to create collection {self.collection_name}",
                    "qdrant",
                    {"error": str(e)}
                )
    
    @retry(
        retry=retry_if_exception_type(ResponseHandlingException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
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
        with tracer.start_as_current_span("qdrant_add_embeddings") as span:
            texts_list = list(texts)
            num_docs = len(texts_list)
            
            span.set_attribute("collection_name", self.collection_name)
            span.set_attribute("num_documents", num_docs)
            
            if len(embeddings) != num_docs:
                raise VectorStoreError(
                    "Number of texts and embeddings must match",
                    "qdrant"
                )
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(num_docs)]
            elif len(ids) != num_docs:
                raise VectorStoreError(
                    "Number of IDs must match number of texts",
                    "qdrant"
                )
            
            # Prepare metadata
            if metadatas is None:
                metadatas = [{} for _ in range(num_docs)]
            elif len(metadatas) != num_docs:
                raise VectorStoreError(
                    "Number of metadatas must match number of texts",
                    "qdrant"
                )
            
            try:
                # Prepare points
                points = []
                for i, (text, embedding, metadata, doc_id) in enumerate(
                    zip(texts_list, embeddings, metadatas, ids)
                ):
                    payload = {
                        "content": text,
                        "metadata": metadata,
                        **metadata  # Flatten metadata for easier filtering
                    }
                    
                    points.append(
                        models.PointStruct(
                            id=doc_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                # Upsert points
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logger.info(
                    f"Added {num_docs} documents to {self.collection_name}",
                    extra={"document_ids": ids[:5]}  # Log first 5 IDs
                )
                
                span.set_attribute("success", True)
                return ids
                
            except Exception as e:
                logger.error(f"Failed to add embeddings: {e}")
                span.set_attribute("error", str(e))
                raise VectorStoreError(
                    f"Failed to add embeddings to {self.collection_name}",
                    "qdrant",
                    {"error": str(e), "num_documents": num_docs}
                )
    
    @retry(
        retry=retry_if_exception_type(ResponseHandlingException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
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
        with tracer.start_as_current_span("qdrant_similarity_search_by_vector") as span:
            span.set_attribute("collection_name", self.collection_name)
            span.set_attribute("k", k)
            span.set_attribute("has_filter", filter is not None)
            
            try:
                # Prepare filter
                qdrant_filter = None
                if filter:
                    conditions = []
                    for key, value in filter.items():
                        if isinstance(value, list):
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value)
                                )
                            )
                        else:
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=value)
                                )
                            )
                    
                    if conditions:
                        qdrant_filter = models.Filter(must=conditions)
                
                # Search
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    limit=k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Convert to documents
                documents = []
                for result in results:
                    payload = result.payload or {}
                    content = payload.get("content", "")
                    metadata = payload.get("metadata", {})
                    
                    doc = Document(
                        id=str(result.id),
                        content=content,
                        metadata=metadata,
                        score=result.score
                    )
                    documents.append(doc)
                
                logger.info(
                    f"Found {len(documents)} similar documents",
                    extra={
                        "collection_name": self.collection_name,
                        "k": k,
                        "results_count": len(documents)
                    }
                )
                
                span.set_attribute("results_count", len(documents))
                return documents
                
            except Exception as e:
                logger.error(f"Failed to search: {e}")
                span.set_attribute("error", str(e))
                raise VectorStoreError(
                    f"Failed to search in {self.collection_name}",
                    "qdrant",
                    {"error": str(e), "k": k}
                )
    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Perform similarity search.
        
        Note: This method requires an embedder to convert query to vector.
        Use similarity_search_by_vector directly if you have the embedding.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        raise NotImplementedError(
            "similarity_search requires an embedder. "
            "Use similarity_search_by_vector with pre-computed embedding."
        )
    
    @retry(
        retry=retry_if_exception_type(ResponseHandlingException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
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
        with tracer.start_as_current_span("qdrant_delete") as span:
            span.set_attribute("collection_name", self.collection_name)
            span.set_attribute("num_ids", len(ids))
            
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=ids
                    )
                )
                
                logger.info(
                    f"Deleted {len(ids)} documents from {self.collection_name}",
                    extra={"document_ids": ids[:5]}  # Log first 5 IDs
                )
                
                span.set_attribute("success", True)
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete documents: {e}")
                span.set_attribute("error", str(e))
                raise VectorStoreError(
                    f"Failed to delete documents from {self.collection_name}",
                    "qdrant",
                    {"error": str(e), "num_ids": len(ids)}
                )
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information.
        
        Returns:
            Collection information
        """
        with tracer.start_as_current_span("qdrant_get_collection_info") as span:
            span.set_attribute("collection_name", self.collection_name)
            
            try:
                info = self.client.get_collection(self.collection_name)
                
                result = {
                    "name": info.config.params.vectors.size,
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,
                    "points_count": info.points_count,
                    "segments_count": info.segments_count,
                    "status": info.status.value
                }
                
                span.set_attribute("points_count", info.points_count)
                return result
                
            except Exception as e:
                logger.error(f"Failed to get collection info: {e}")
                span.set_attribute("error", str(e))
                raise VectorStoreError(
                    f"Failed to get info for {self.collection_name}",
                    "qdrant",
                    {"error": str(e)}
                )
    
    async def close(self) -> None:
        """Close the client connection."""
        try:
            self.client.close()
            logger.info("Qdrant client closed")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")
