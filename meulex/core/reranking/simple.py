"""Simple reranker implementations."""

import logging
import re
from typing import List

from meulex.config.settings import Settings
from meulex.core.reranking.base import BaseReranker
from meulex.core.vector.base import Document
from meulex.observability import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class KeywordReranker(BaseReranker):
    """Simple keyword-based reranker."""
    
    def __init__(self, settings: Settings):
        """Initialize keyword reranker.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.boost_factor = 1.5  # Boost factor for keyword matches
        
        logger.info("Keyword reranker initialized")
    
    @property
    def name(self) -> str:
        """Get the reranker name."""
        return "keyword"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - convert to lowercase and split
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_keyword_score(self, query_keywords: List[str], document: Document) -> float:
        """Calculate keyword match score.
        
        Args:
            query_keywords: Keywords from query
            document: Document to score
            
        Returns:
            Keyword match score
        """
        doc_text = document.content.lower()
        doc_keywords = set(self._extract_keywords(document.content))
        
        # Count exact keyword matches
        exact_matches = sum(1 for keyword in query_keywords if keyword in doc_keywords)
        
        # Count partial matches in content
        partial_matches = sum(1 for keyword in query_keywords if keyword in doc_text)
        
        # Calculate score based on matches
        if not query_keywords:
            return 0.0
        
        exact_score = exact_matches / len(query_keywords)
        partial_score = partial_matches / len(query_keywords)
        
        # Combine scores with weights
        return (exact_score * 0.7) + (partial_score * 0.3)
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """Rerank documents based on keyword relevance.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents with updated scores
        """
        with tracer.start_as_current_span("keyword_rerank") as span:
            span.set_attribute("query_length", len(query))
            span.set_attribute("document_count", len(documents))
            span.set_attribute("top_k", top_k)
            
            if not documents:
                return []
            
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            span.set_attribute("query_keywords", len(query_keywords))
            
            # Calculate keyword scores and combine with original scores
            reranked_docs = []
            
            for doc in documents:
                keyword_score = self._calculate_keyword_score(query_keywords, doc)
                
                # Combine original score with keyword score
                original_score = doc.score or 0.0
                combined_score = original_score + (keyword_score * self.boost_factor)
                
                # Create new document with updated score
                reranked_doc = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding,
                    score=combined_score
                )
                
                # Add reranking metadata
                reranked_doc.metadata["rerank_keyword_score"] = keyword_score
                reranked_doc.metadata["rerank_original_score"] = original_score
                reranked_doc.metadata["reranker"] = self.name
                
                reranked_docs.append(reranked_doc)
            
            # Sort by combined score
            reranked_docs.sort(key=lambda x: x.score or 0.0, reverse=True)
            
            # Return top-k
            result = reranked_docs[:top_k]
            
            logger.info(
                f"Keyword reranking: {len(documents)} → {len(result)} docs",
                extra={
                    "query_keywords": len(query_keywords),
                    "input_docs": len(documents),
                    "output_docs": len(result),
                    "top_score": result[0].score if result else 0.0
                }
            )
            
            span.set_attribute("output_docs", len(result))
            span.set_attribute("success", True)
            
            return result
    
    async def close(self) -> None:
        """Close the reranker (no-op for keyword reranker)."""
        logger.info("Keyword reranker closed")


class MockReranker(BaseReranker):
    """Mock reranker for development/testing."""
    
    def __init__(self, settings: Settings):
        """Initialize mock reranker.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        logger.info("Mock reranker initialized")
    
    @property
    def name(self) -> str:
        """Get the reranker name."""
        return "mock"
    
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """Mock reranking - just returns documents with slight score adjustments.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Documents with mock reranking scores
        """
        with tracer.start_as_current_span("mock_rerank") as span:
            span.set_attribute("document_count", len(documents))
            span.set_attribute("top_k", top_k)
            
            if not documents:
                return []
            
            # Apply small random adjustments to scores
            import random
            reranked_docs = []
            
            for i, doc in enumerate(documents):
                # Small random adjustment to simulate reranking
                adjustment = random.uniform(-0.1, 0.1)
                original_score = doc.score or 0.0
                new_score = max(0.0, original_score + adjustment)
                
                reranked_doc = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding,
                    score=new_score
                )
                
                # Add mock reranking metadata
                reranked_doc.metadata["rerank_adjustment"] = adjustment
                reranked_doc.metadata["rerank_original_score"] = original_score
                reranked_doc.metadata["reranker"] = self.name
                
                reranked_docs.append(reranked_doc)
            
            # Sort by new scores
            reranked_docs.sort(key=lambda x: x.score or 0.0, reverse=True)
            
            result = reranked_docs[:top_k]
            
            logger.info(f"Mock reranking: {len(documents)} → {len(result)} docs")
            
            span.set_attribute("output_docs", len(result))
            span.set_attribute("success", True)
            
            return result
    
    async def close(self) -> None:
        """Close the mock reranker (no-op)."""
        logger.info("Mock reranker closed")
