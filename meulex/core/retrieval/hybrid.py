"""Hybrid retrieval combining dense and sparse methods with RRF fusion."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from meulex.config.settings import Settings
from meulex.core.embeddings.base import BaseEmbedder
from meulex.core.retrieval.dense import DenseRetriever
from meulex.core.retrieval.sparse import BM25Retriever
from meulex.core.vector.base import Document, VectorStore
from meulex.observability import DOCUMENTS_RETRIEVED, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval with RRF fusion."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        settings: Settings
    ):
        """Initialize hybrid retriever.
        
        Args:
            embedder: Embeddings provider
            vector_store: Vector store
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize dense retriever
        self.dense_retriever = DenseRetriever(embedder, vector_store, settings)
        
        # Initialize sparse retriever if enabled
        self.sparse_retriever = None
        if settings.enable_sparse_retrieval:
            self.sparse_retriever = BM25Retriever(settings)
            logger.info("Sparse retrieval enabled")
        
        # RRF parameters
        self.rrf_k = settings.rrf_k
        self.dense_weight = settings.dense_weight
        self.sparse_weight = settings.sparse_weight
        
        logger.info(
            "Hybrid retriever initialized",
            extra={
                "dense_enabled": True,
                "sparse_enabled": settings.enable_sparse_retrieval,
                "rrf_k": self.rrf_k,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight
            }
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both dense and sparse indexes.
        
        Args:
            documents: List of documents to add
        """
        with tracer.start_as_current_span("hybrid_add_documents") as span:
            span.set_attribute("document_count", len(documents))
            
            # Add to sparse retriever if enabled
            if self.sparse_retriever:
                self.sparse_retriever.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to sparse index")
            
            span.set_attribute("success", True)
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        k: int = 60
    ) -> List[Document]:
        """Apply Reciprocal Rank Fusion to combine results.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: RRF parameter (default 60)
            
        Returns:
            Fused and ranked results
        """
        with tracer.start_as_current_span("rrf_fusion") as span:
            span.set_attribute("dense_count", len(dense_results))
            span.set_attribute("sparse_count", len(sparse_results))
            span.set_attribute("rrf_k", k)
            
            # Create document ID to document mapping
            doc_map = {}
            
            # RRF scores for each document
            rrf_scores = defaultdict(float)
            
            # Process dense results
            for rank, doc in enumerate(dense_results, 1):
                doc_id = doc.id
                doc_map[doc_id] = doc
                rrf_score = self.dense_weight / (k + rank)
                rrf_scores[doc_id] += rrf_score
            
            # Process sparse results
            for rank, doc in enumerate(sparse_results, 1):
                doc_id = doc.id
                doc_map[doc_id] = doc
                rrf_score = self.sparse_weight / (k + rank)
                rrf_scores[doc_id] += rrf_score
            
            # Sort by RRF score
            sorted_docs = sorted(
                rrf_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create result list with RRF scores
            results = []
            for doc_id, rrf_score in sorted_docs:
                doc = doc_map[doc_id]
                # Create a copy with the RRF score
                result_doc = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding,
                    score=rrf_score
                )
                results.append(result_doc)
            
            logger.info(
                f"RRF fusion combined {len(dense_results)} dense + {len(sparse_results)} sparse → {len(results)} unique results",
                extra={
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "unique_results": len(results),
                    "top_rrf_score": results[0].score if results else 0.0
                }
            )
            
            span.set_attribute("unique_results", len(results))
            span.set_attribute("success", True)
            
            return results
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid search with RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of retrieved documents with RRF scores
        """
        with tracer.start_as_current_span("hybrid_retrieve") as span:
            top_k = top_k or self.settings.default_top_k
            
            span.set_attribute("query_length", len(query))
            span.set_attribute("top_k", top_k)
            span.set_attribute("has_filter", filter is not None)
            
            # Retrieve more documents from each method for better fusion
            # Use 2x top_k to ensure good coverage
            retrieval_k = min(top_k * 2, self.settings.max_top_k)
            
            results = []
            
            try:
                # Dense retrieval
                dense_results = await self.dense_retriever.retrieve(
                    query=query,
                    top_k=retrieval_k,
                    filter=filter
                )
                
                span.set_attribute("dense_results", len(dense_results))
                
                # Sparse retrieval (if enabled)
                sparse_results = []
                if self.sparse_retriever and self.settings.enable_sparse_retrieval:
                    sparse_results = self.sparse_retriever.retrieve(
                        query=query,
                        top_k=retrieval_k,
                        filter=filter
                    )
                    
                    span.set_attribute("sparse_results", len(sparse_results))
                
                # Combine results using RRF
                if sparse_results:
                    results = self._reciprocal_rank_fusion(
                        dense_results,
                        sparse_results,
                        self.rrf_k
                    )
                else:
                    # Only dense results available
                    results = dense_results
                
                # Limit to requested top_k
                results = results[:top_k]
                
                # Record metrics
                DOCUMENTS_RETRIEVED.labels(strategy="hybrid").inc(len(results))
                
                logger.info(
                    f"Hybrid retrieval returned {len(results)} documents",
                    extra={
                        "query_length": len(query),
                        "dense_results": len(dense_results),
                        "sparse_results": len(sparse_results),
                        "final_results": len(results),
                        "fusion_used": len(sparse_results) > 0
                    }
                )
                
                span.set_attribute("final_results", len(results))
                span.set_attribute("fusion_used", len(sparse_results) > 0)
                span.set_attribute("success", True)
                
                return results
                
            except Exception as e:
                logger.error(f"Hybrid retrieval failed: {e}")
                span.set_attribute("error", str(e))
                raise
    
    def get_stats(self) -> Dict:
        """Get retriever statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "type": "hybrid",
            "dense_retriever": {
                "embedder": self.dense_retriever.embedder.model_name,
                "vector_store": type(self.dense_retriever.vector_store).__name__
            },
            "sparse_retriever": None,
            "fusion": {
                "rrf_k": self.rrf_k,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight
            }
        }
        
        if self.sparse_retriever:
            stats["sparse_retriever"] = self.sparse_retriever.get_stats()
        
        return stats
    
    async def close(self) -> None:
        """Close the hybrid retriever."""
        try:
            await self.dense_retriever.close()
            logger.info("Hybrid retriever closed")
        except Exception as e:
            logger.warning(f"Error closing hybrid retriever: {e}")
