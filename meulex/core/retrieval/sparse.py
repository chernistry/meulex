"""Sparse retrieval using BM25 algorithm."""

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from meulex.config.settings import Settings
from meulex.core.vector.base import Document
from meulex.observability import DOCUMENTS_RETRIEVED, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class BM25Retriever:
    """BM25-based sparse retriever."""
    
    def __init__(self, settings: Settings, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 retriever.
        
        Args:
            settings: Application settings
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.settings = settings
        self.k1 = k1
        self.b = b
        
        # Document storage
        self.documents: List[Document] = []
        self.doc_frequencies: Dict[str, int] = defaultdict(int)  # term -> doc count
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        
        # Inverted index: term -> list of (doc_id, term_frequency)
        self.inverted_index: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        
        logger.info(
            "BM25 retriever initialized",
            extra={"k1": k1, "b": b}
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - in production, use a proper tokenizer
        import re
        
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the BM25 index.
        
        Args:
            documents: List of documents to add
        """
        with tracer.start_as_current_span("bm25_add_documents") as span:
            span.set_attribute("document_count", len(documents))
            
            start_doc_id = len(self.documents)
            
            for i, doc in enumerate(documents):
                doc_id = start_doc_id + i
                self.documents.append(doc)
                
                # Tokenize document content
                tokens = self._tokenize(doc.content)
                doc_length = len(tokens)
                self.doc_lengths.append(doc_length)
                
                # Count term frequencies in this document
                term_frequencies = Counter(tokens)
                
                # Update inverted index and document frequencies
                for term, tf in term_frequencies.items():
                    self.inverted_index[term].append((doc_id, tf))
                    if tf > 0:  # First occurrence of term in this doc
                        self.doc_frequencies[term] += 1
            
            # Update statistics
            self.total_docs = len(self.documents)
            self.avg_doc_length = sum(self.doc_lengths) / self.total_docs if self.total_docs > 0 else 0
            
            logger.info(
                f"Added {len(documents)} documents to BM25 index",
                extra={
                    "total_docs": self.total_docs,
                    "avg_doc_length": self.avg_doc_length,
                    "vocabulary_size": len(self.inverted_index)
                }
            )
            
            span.set_attribute("total_docs", self.total_docs)
            span.set_attribute("vocabulary_size", len(self.inverted_index))
    
    def _calculate_bm25_score(
        self,
        query_terms: List[str],
        doc_id: int,
        doc_term_frequencies: Dict[str, int]
    ) -> float:
        """Calculate BM25 score for a document.
        
        Args:
            query_terms: Query terms
            doc_id: Document ID
            doc_term_frequencies: Term frequencies in the document
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        for term in query_terms:
            if term in doc_term_frequencies:
                tf = doc_term_frequencies[term]
                df = self.doc_frequencies[term]
                
                # IDF calculation
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter: Optional metadata filter (not implemented)
            
        Returns:
            List of retrieved documents with BM25 scores
        """
        with tracer.start_as_current_span("bm25_retrieve") as span:
            top_k = top_k or self.settings.default_top_k
            
            span.set_attribute("query_length", len(query))
            span.set_attribute("top_k", top_k)
            span.set_attribute("total_docs", self.total_docs)
            
            if self.total_docs == 0:
                logger.warning("No documents in BM25 index")
                return []
            
            # Tokenize query
            query_terms = self._tokenize(query)
            if not query_terms:
                logger.warning("No valid query terms after tokenization")
                return []
            
            span.set_attribute("query_terms", len(query_terms))
            
            # Find candidate documents
            candidate_docs: Set[int] = set()
            for term in query_terms:
                if term in self.inverted_index:
                    for doc_id, _ in self.inverted_index[term]:
                        candidate_docs.add(doc_id)
            
            if not candidate_docs:
                logger.info("No candidate documents found for query terms")
                return []
            
            span.set_attribute("candidate_docs", len(candidate_docs))
            
            # Calculate BM25 scores for candidate documents
            doc_scores = []
            
            for doc_id in candidate_docs:
                # Get term frequencies for this document
                doc_term_frequencies = {}
                for term in query_terms:
                    if term in self.inverted_index:
                        for d_id, tf in self.inverted_index[term]:
                            if d_id == doc_id:
                                doc_term_frequencies[term] = tf
                                break
                
                # Calculate BM25 score
                score = self._calculate_bm25_score(query_terms, doc_id, doc_term_frequencies)
                
                if score > 0:
                    doc_scores.append((doc_id, score))
            
            # Sort by score and take top-k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = doc_scores[:top_k]
            
            # Create result documents with scores
            results = []
            for doc_id, score in top_docs:
                doc = self.documents[doc_id]
                # Create a copy with the score
                result_doc = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding,
                    score=score
                )
                results.append(result_doc)
            
            # Record metrics
            DOCUMENTS_RETRIEVED.labels(strategy="sparse").inc(len(results))
            
            logger.info(
                f"BM25 retrieved {len(results)} documents",
                extra={
                    "query_terms": len(query_terms),
                    "candidate_docs": len(candidate_docs),
                    "results_count": len(results),
                    "top_score": results[0].score if results else 0.0
                }
            )
            
            span.set_attribute("results_count", len(results))
            span.set_attribute("success", True)
            
            return results
    
    def get_stats(self) -> Dict:
        """Get retriever statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_documents": self.total_docs,
            "vocabulary_size": len(self.inverted_index),
            "average_document_length": self.avg_doc_length,
            "parameters": {
                "k1": self.k1,
                "b": self.b
            }
        }
