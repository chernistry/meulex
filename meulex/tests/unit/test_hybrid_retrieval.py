"""Unit tests for hybrid retrieval functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from meulex.config.settings import Settings
from meulex.core.retrieval.hybrid import HybridRetriever
from meulex.core.retrieval.sparse import BM25Retriever
from meulex.core.vector.base import Document


class TestBM25Retriever:
    """Test BM25 sparse retriever functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock(spec=Settings)
        settings.default_top_k = 3
        return settings
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Python is a programming language used for web development and data science.",
                metadata={"source": "python_guide.txt"}
            ),
            Document(
                id="doc2", 
                content="Machine learning algorithms can be implemented in Python using libraries like scikit-learn.",
                metadata={"source": "ml_guide.txt"}
            ),
            Document(
                id="doc3",
                content="Web development with Python frameworks like Django and Flask is very popular.",
                metadata={"source": "web_guide.txt"}
            )
        ]
    
    def test_bm25_initialization(self, mock_settings):
        """Test BM25 retriever initialization."""
        retriever = BM25Retriever(mock_settings)
        
        assert retriever.settings == mock_settings
        assert retriever.k1 == 1.2
        assert retriever.b == 0.75
        assert retriever.total_docs == 0
        assert len(retriever.documents) == 0
    
    def test_bm25_tokenization(self, mock_settings):
        """Test BM25 tokenization functionality."""
        retriever = BM25Retriever(mock_settings)
        
        text = "Python is a great programming language for data science!"
        tokens = retriever._tokenize(text)
        
        # Should remove stop words and short tokens
        expected_tokens = ["python", "great", "programming", "language", "data", "science"]
        assert tokens == expected_tokens
    
    def test_bm25_add_documents(self, mock_settings, sample_documents):
        """Test adding documents to BM25 index."""
        retriever = BM25Retriever(mock_settings)
        
        retriever.add_documents(sample_documents)
        
        assert retriever.total_docs == 3
        assert len(retriever.documents) == 3
        assert len(retriever.inverted_index) > 0
        assert retriever.avg_doc_length > 0
        
        # Check that terms are indexed
        assert "python" in retriever.inverted_index
        assert "programming" in retriever.inverted_index
    
    def test_bm25_scoring(self, mock_settings, sample_documents):
        """Test BM25 scoring functionality."""
        retriever = BM25Retriever(mock_settings)
        retriever.add_documents(sample_documents)
        
        query_terms = ["python", "programming"]
        doc_id = 0  # First document
        
        # Get term frequencies for the document
        doc_term_frequencies = {}
        for term in query_terms:
            if term in retriever.inverted_index:
                for d_id, tf in retriever.inverted_index[term]:
                    if d_id == doc_id:
                        doc_term_frequencies[term] = tf
                        break
        
        score = retriever._calculate_bm25_score(query_terms, doc_id, doc_term_frequencies)
        assert score >= 0.0  # BM25 scores should be non-negative
    
    def test_bm25_retrieve(self, mock_settings, sample_documents):
        """Test BM25 document retrieval."""
        retriever = BM25Retriever(mock_settings)
        retriever.add_documents(sample_documents)
        
        # Query for Python-related content
        results = retriever.retrieve("python programming", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        assert all(doc.score is not None for doc in results)
        
        # Results should be sorted by score (descending)
        if len(results) > 1:
            assert results[0].score >= results[1].score
    
    def test_bm25_retrieve_empty_index(self, mock_settings):
        """Test BM25 retrieval with empty index."""
        retriever = BM25Retriever(mock_settings)
        
        results = retriever.retrieve("any query")
        assert results == []
    
    def test_bm25_retrieve_no_matches(self, mock_settings, sample_documents):
        """Test BM25 retrieval with no matching documents."""
        retriever = BM25Retriever(mock_settings)
        retriever.add_documents(sample_documents)
        
        # Query for content not in documents
        results = retriever.retrieve("quantum physics nuclear")
        
        # Should return empty list or documents with very low scores
        assert len(results) == 0 or all(doc.score < 0.1 for doc in results)
    
    def test_bm25_stats(self, mock_settings, sample_documents):
        """Test BM25 statistics reporting."""
        retriever = BM25Retriever(mock_settings)
        retriever.add_documents(sample_documents)
        
        stats = retriever.get_stats()
        
        assert stats["total_documents"] == 3
        assert stats["vocabulary_size"] > 0
        assert stats["average_document_length"] > 0
        assert stats["parameters"]["k1"] == 1.2
        assert stats["parameters"]["b"] == 0.75


class TestHybridRetriever:
    """Test hybrid retrieval functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock(spec=Settings)
        settings.enable_sparse_retrieval = True
        settings.enable_reranker = False
        settings.rrf_k = 60
        settings.dense_weight = 1.0
        settings.sparse_weight = 1.0
        settings.default_top_k = 3
        settings.max_top_k = 20
        return settings
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = AsyncMock()
        embedder.model_name = "mock-embedder"
        return embedder
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = AsyncMock()
        return store
    
    @pytest.fixture
    def mock_dense_retriever(self):
        """Create mock dense retriever."""
        retriever = AsyncMock()
        retriever.embedder.model_name = "mock-embedder"
        retriever.vector_store = MagicMock()
        return retriever
    
    @pytest.fixture
    def sample_dense_results(self):
        """Create sample dense retrieval results."""
        return [
            Document(
                id="dense1",
                content="Dense result 1",
                metadata={"source": "dense1.txt"},
                score=0.9
            ),
            Document(
                id="dense2",
                content="Dense result 2", 
                metadata={"source": "dense2.txt"},
                score=0.8
            )
        ]
    
    @pytest.fixture
    def sample_sparse_results(self):
        """Create sample sparse retrieval results."""
        return [
            Document(
                id="sparse1",
                content="Sparse result 1",
                metadata={"source": "sparse1.txt"},
                score=2.5
            ),
            Document(
                id="dense1",  # Overlap with dense results
                content="Dense result 1",
                metadata={"source": "dense1.txt"},
                score=2.0
            )
        ]
    
    @patch('meulex.core.retrieval.hybrid.DenseRetriever')
    @patch('meulex.core.retrieval.hybrid.BM25Retriever')
    def test_hybrid_initialization(
        self, 
        mock_bm25_class, 
        mock_dense_class,
        mock_settings, 
        mock_embedder, 
        mock_vector_store
    ):
        """Test hybrid retriever initialization."""
        mock_dense_instance = MagicMock()
        mock_dense_class.return_value = mock_dense_instance
        
        mock_sparse_instance = MagicMock()
        mock_bm25_class.return_value = mock_sparse_instance
        
        retriever = HybridRetriever(mock_embedder, mock_vector_store, mock_settings)
        
        assert retriever.dense_retriever == mock_dense_instance
        assert retriever.sparse_retriever == mock_sparse_instance
        assert retriever.rrf_k == 60
        assert retriever.dense_weight == 1.0
        assert retriever.sparse_weight == 1.0
    
    def test_rrf_fusion(self, mock_settings, mock_embedder, mock_vector_store):
        """Test Reciprocal Rank Fusion algorithm."""
        with patch('meulex.core.retrieval.hybrid.DenseRetriever'), \
             patch('meulex.core.retrieval.hybrid.BM25Retriever'):
            
            retriever = HybridRetriever(mock_embedder, mock_vector_store, mock_settings)
            
            dense_results = [
                Document(id="doc1", content="Content 1", score=0.9),
                Document(id="doc2", content="Content 2", score=0.8)
            ]
            
            sparse_results = [
                Document(id="doc2", content="Content 2", score=2.0),  # Overlap
                Document(id="doc3", content="Content 3", score=1.5)
            ]
            
            fused_results = retriever._reciprocal_rank_fusion(
                dense_results, sparse_results, k=60
            )
            
            # Should have 3 unique documents
            assert len(fused_results) == 3
            
            # All documents should have RRF scores
            assert all(doc.score is not None for doc in fused_results)
            
            # Results should be sorted by RRF score
            scores = [doc.score for doc in fused_results]
            assert scores == sorted(scores, reverse=True)
            
            # doc2 should have highest score (appears in both lists)
            doc_ids = [doc.id for doc in fused_results]
            assert doc_ids[0] == "doc2"
    
    @patch('meulex.core.retrieval.hybrid.DenseRetriever')
    @patch('meulex.core.retrieval.hybrid.BM25Retriever')
    async def test_hybrid_retrieve_with_fusion(
        self,
        mock_bm25_class,
        mock_dense_class,
        mock_settings,
        mock_embedder,
        mock_vector_store,
        sample_dense_results,
        sample_sparse_results
    ):
        """Test hybrid retrieval with RRF fusion."""
        # Setup mock dense retriever
        mock_dense_instance = AsyncMock()
        mock_dense_instance.retrieve.return_value = sample_dense_results
        mock_dense_instance.embedder.model_name = "mock-embedder"
        mock_dense_instance.vector_store = mock_vector_store
        mock_dense_class.return_value = mock_dense_instance
        
        # Setup mock sparse retriever
        mock_sparse_instance = MagicMock()
        mock_sparse_instance.retrieve.return_value = sample_sparse_results
        mock_bm25_class.return_value = mock_sparse_instance
        
        retriever = HybridRetriever(mock_embedder, mock_vector_store, mock_settings)
        
        results = await retriever.retrieve("test query", top_k=3)
        
        # Should call both retrievers
        mock_dense_instance.retrieve.assert_called_once()
        mock_sparse_instance.retrieve.assert_called_once()
        
        # Should return fused results
        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)
        assert all(doc.score is not None for doc in results)
    
    @patch('meulex.core.retrieval.hybrid.DenseRetriever')
    async def test_hybrid_retrieve_dense_only(
        self,
        mock_dense_class,
        mock_embedder,
        mock_vector_store,
        sample_dense_results
    ):
        """Test hybrid retrieval with dense only (sparse disabled)."""
        settings = MagicMock(spec=Settings)
        settings.enable_sparse_retrieval = False
        settings.enable_reranker = False
        settings.default_top_k = 3
        settings.max_top_k = 20
        settings.rrf_k = 60
        settings.dense_weight = 1.0
        settings.sparse_weight = 1.0
        
        # Setup mock dense retriever
        mock_dense_instance = AsyncMock()
        mock_dense_instance.retrieve.return_value = sample_dense_results
        mock_dense_instance.embedder.model_name = "mock-embedder"
        mock_dense_instance.vector_store = mock_vector_store
        mock_dense_class.return_value = mock_dense_instance
        
        retriever = HybridRetriever(mock_embedder, mock_vector_store, settings)
        
        results = await retriever.retrieve("test query", top_k=2)
        
        # Should only call dense retriever
        mock_dense_instance.retrieve.assert_called_once()
        
        # Should return dense results directly
        assert results == sample_dense_results
    
    @patch('meulex.core.retrieval.hybrid.DenseRetriever')
    @patch('meulex.core.retrieval.hybrid.BM25Retriever')
    @patch('meulex.core.retrieval.hybrid.create_reranker')
    async def test_hybrid_retrieve_with_reranker(
        self,
        mock_create_reranker,
        mock_bm25_class,
        mock_dense_class,
        mock_embedder,
        mock_vector_store,
        sample_dense_results
    ):
        """Test hybrid retrieval with reranking enabled."""
        settings = MagicMock(spec=Settings)
        settings.enable_sparse_retrieval = False
        settings.enable_reranker = True
        settings.default_top_k = 3
        settings.max_top_k = 20
        settings.rrf_k = 60
        settings.dense_weight = 1.0
        settings.sparse_weight = 1.0
        
        # Setup mock reranker
        mock_reranker = AsyncMock()
        mock_reranker.name = "keyword"
        reranked_results = [
            Document(id="reranked1", content="Reranked 1", score=1.5),
            Document(id="reranked2", content="Reranked 2", score=1.2)
        ]
        mock_reranker.rerank.return_value = reranked_results
        mock_create_reranker.return_value = mock_reranker
        
        # Setup mock dense retriever
        mock_dense_instance = AsyncMock()
        mock_dense_instance.retrieve.return_value = sample_dense_results
        mock_dense_instance.embedder.model_name = "mock-embedder"
        mock_dense_instance.vector_store = mock_vector_store
        mock_dense_class.return_value = mock_dense_instance
        
        retriever = HybridRetriever(mock_embedder, mock_vector_store, settings)
        
        results = await retriever.retrieve("test query", top_k=2)
        
        # Should call reranker
        mock_reranker.rerank.assert_called_once()
        
        # Should return reranked results
        assert results == reranked_results
    
    @patch('meulex.core.retrieval.hybrid.DenseRetriever')
    @patch('meulex.core.retrieval.hybrid.BM25Retriever')
    def test_hybrid_stats(
        self,
        mock_bm25_class,
        mock_dense_class,
        mock_settings,
        mock_embedder,
        mock_vector_store
    ):
        """Test hybrid retriever statistics."""
        mock_dense_instance = MagicMock()
        mock_dense_instance.embedder.model_name = "mock-embedder"
        mock_dense_instance.vector_store = mock_vector_store
        mock_dense_class.return_value = mock_dense_instance
        
        mock_sparse_instance = MagicMock()
        mock_sparse_instance.get_stats.return_value = {"total_documents": 10}
        mock_bm25_class.return_value = mock_sparse_instance
        
        retriever = HybridRetriever(mock_embedder, mock_vector_store, mock_settings)
        
        stats = retriever.get_stats()
        
        assert stats["type"] == "hybrid"
        assert stats["dense_retriever"]["embedder"] == "mock-embedder"
        assert stats["sparse_retriever"]["total_documents"] == 10
        assert stats["fusion"]["rrf_k"] == 60
        assert stats["fusion"]["dense_weight"] == 1.0
        assert stats["fusion"]["sparse_weight"] == 1.0
    
    @patch('meulex.core.retrieval.hybrid.DenseRetriever')
    @patch('meulex.core.retrieval.hybrid.BM25Retriever')
    async def test_hybrid_cleanup(
        self,
        mock_bm25_class,
        mock_dense_class,
        mock_settings,
        mock_embedder,
        mock_vector_store
    ):
        """Test hybrid retriever cleanup."""
        mock_dense_instance = AsyncMock()
        mock_dense_class.return_value = mock_dense_instance
        
        retriever = HybridRetriever(mock_embedder, mock_vector_store, mock_settings)
        
        await retriever.close()
        
        mock_dense_instance.close.assert_called_once()


@pytest.mark.asyncio
class TestHybridRetrievalIntegration:
    """Integration tests for hybrid retrieval."""
    
    async def test_hybrid_with_mock_components(self):
        """Test hybrid retrieval with mock components."""
        settings = Settings(
            enable_sparse_retrieval=True,
            enable_reranker=False,
            default_top_k=3,
            rrf_k=60,
            dense_weight=1.0,
            sparse_weight=1.0
        )
        
        # Create mock embedder and vector store
        embedder = AsyncMock()
        embedder.model_name = "mock-embedder"
        
        vector_store = AsyncMock()
        
        # This would require more complex mocking for full integration
        # For now, we test the initialization
        with patch('meulex.core.retrieval.hybrid.DenseRetriever'), \
             patch('meulex.core.retrieval.hybrid.BM25Retriever'):
            
            retriever = HybridRetriever(embedder, vector_store, settings)
            
            assert retriever.settings == settings
            assert retriever.sparse_retriever is not None
            
            await retriever.close()
