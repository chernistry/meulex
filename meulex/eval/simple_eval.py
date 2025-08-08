"""Simple RAG evaluation framework for Meulex."""

import json
import logging
import time
from typing import Dict, List, Optional

from meulex.config.settings import Settings
from meulex.core.embeddings.factory import create_embedder
from meulex.core.ingestion.ingestor import DocumentIngestor
from meulex.core.retrieval.hybrid import HybridRetriever
from meulex.core.vector.qdrant_store import QdrantStore
from meulex.llm.cascade import LLMCascade
from meulex.llm.prompt_builder import RAGPromptBuilder

logger = logging.getLogger(__name__)


class SimpleRAGEvaluator:
    """Simple RAG evaluation framework."""
    
    def __init__(self, settings: Settings):
        """Initialize RAG evaluator.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Sample evaluation dataset
        self.eval_dataset = [
            {
                "question": "What is Python used for?",
                "expected_keywords": ["programming", "development", "language"],
                "context": "Python is a versatile programming language used for web development, data science, automation, and machine learning."
            },
            {
                "question": "How does machine learning work?",
                "expected_keywords": ["algorithms", "data", "models", "training"],
                "context": "Machine learning uses algorithms to analyze data, identify patterns, and make predictions by training models on historical data."
            },
            {
                "question": "What are the benefits of FastAPI?",
                "expected_keywords": ["fast", "api", "documentation", "python"],
                "context": "FastAPI is a modern, fast web framework for building APIs with Python, featuring automatic documentation generation and high performance."
            }
        ]
    
    async def setup_test_environment(self) -> tuple:
        """Setup test environment with sample data.
        
        Returns:
            Tuple of (retriever, llm_cascade, prompt_builder)
        """
        # Initialize components
        embedder = create_embedder(self.settings)
        vector_store = QdrantStore(self.settings)
        await vector_store.create_collection()
        
        # Initialize retriever and LLM
        retriever = HybridRetriever(embedder, vector_store, self.settings)
        llm_cascade = LLMCascade(self.settings)
        prompt_builder = RAGPromptBuilder()
        
        # Ingest sample documents
        ingestor = DocumentIngestor(embedder, vector_store, self.settings)
        
        sample_docs = [
            {
                "content": item["context"],
                "metadata": {"source": f"eval_doc_{i}.txt", "type": "evaluation"}
            }
            for i, item in enumerate(self.eval_dataset)
        ]
        
        for doc in sample_docs:
            await ingestor.ingest_document(doc["content"], doc["metadata"])
        
        # Add documents to sparse index if enabled
        if self.settings.enable_sparse_retrieval and retriever.sparse_retriever:
            from meulex.core.vector.base import Document
            documents = [
                Document(
                    id=str(i),
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
                for i, doc in enumerate(sample_docs)
            ]
            retriever.sparse_retriever.add_documents(documents)
        
        logger.info(f"Setup test environment with {len(sample_docs)} documents")
        return retriever, llm_cascade, prompt_builder
    
    def calculate_keyword_overlap(self, text: str, expected_keywords: List[str]) -> float:
        """Calculate keyword overlap score.
        
        Args:
            text: Generated text
            expected_keywords: Expected keywords
            
        Returns:
            Overlap score (0.0 to 1.0)
        """
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in expected_keywords if keyword in text_lower)
        return found_keywords / len(expected_keywords) if expected_keywords else 0.0
    
    def calculate_relevance_score(self, sources: List[Dict], question: str) -> float:
        """Calculate relevance score based on retrieved sources.
        
        Args:
            sources: Retrieved source documents
            question: Original question
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not sources:
            return 0.0
        
        # Simple relevance based on average source scores
        scores = [source.get("score", 0.0) for source in sources]
        return sum(scores) / len(scores) if scores else 0.0
    
    def calculate_response_quality(self, answer: str) -> Dict[str, float]:
        """Calculate response quality metrics.
        
        Args:
            answer: Generated answer
            
        Returns:
            Quality metrics dictionary
        """
        return {
            "length": len(answer),
            "word_count": len(answer.split()),
            "has_content": 1.0 if len(answer.strip()) > 0 else 0.0,
            "completeness": min(1.0, len(answer) / 100)  # Assume 100 chars is complete
        }
    
    async def evaluate_single_query(
        self,
        retriever: HybridRetriever,
        llm_cascade: LLMCascade,
        prompt_builder: RAGPromptBuilder,
        eval_item: Dict
    ) -> Dict:
        """Evaluate a single query.
        
        Args:
            retriever: Document retriever
            llm_cascade: LLM cascade
            prompt_builder: Prompt builder
            eval_item: Evaluation item
            
        Returns:
            Evaluation results
        """
        question = eval_item["question"]
        expected_keywords = eval_item["expected_keywords"]
        
        start_time = time.time()
        
        try:
            # Retrieve documents
            documents = await retriever.retrieve(question, top_k=3)
            
            # Generate response
            messages = prompt_builder.build_rag_prompt(
                question=question,
                documents=documents,
                conversation_history=None
            )
            
            llm_response = await llm_cascade.chat_completion(messages)
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            keyword_overlap = self.calculate_keyword_overlap(
                llm_response.content, expected_keywords
            )
            
            sources = [
                {
                    "text": doc.content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.score or 0.0
                }
                for doc in documents
            ]
            
            relevance_score = self.calculate_relevance_score(sources, question)
            quality_metrics = self.calculate_response_quality(llm_response.content)
            
            return {
                "question": question,
                "answer": llm_response.content,
                "sources_count": len(documents),
                "processing_time": processing_time,
                "metrics": {
                    "keyword_overlap": keyword_overlap,
                    "relevance_score": relevance_score,
                    "response_quality": quality_metrics,
                    "retrieval_success": 1.0 if documents else 0.0
                },
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed for question '{question}': {e}")
            return {
                "question": question,
                "answer": "",
                "sources_count": 0,
                "processing_time": time.time() - start_time,
                "metrics": {
                    "keyword_overlap": 0.0,
                    "relevance_score": 0.0,
                    "response_quality": {"length": 0, "word_count": 0, "has_content": 0.0, "completeness": 0.0},
                    "retrieval_success": 0.0
                },
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    async def run_evaluation(self) -> Dict:
        """Run complete evaluation.
        
        Returns:
            Evaluation results
        """
        logger.info("Starting RAG evaluation")
        
        # Setup test environment
        retriever, llm_cascade, prompt_builder = await self.setup_test_environment()
        
        # Run evaluation on each item
        results = []
        for eval_item in self.eval_dataset:
            result = await self.evaluate_single_query(
                retriever, llm_cascade, prompt_builder, eval_item
            )
            results.append(result)
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            avg_metrics = {
                "keyword_overlap": sum(r["metrics"]["keyword_overlap"] for r in successful_results) / len(successful_results),
                "relevance_score": sum(r["metrics"]["relevance_score"] for r in successful_results) / len(successful_results),
                "retrieval_success": sum(r["metrics"]["retrieval_success"] for r in successful_results) / len(successful_results),
                "avg_processing_time": sum(r["processing_time"] for r in successful_results) / len(successful_results),
                "avg_sources_count": sum(r["sources_count"] for r in successful_results) / len(successful_results)
            }
        else:
            avg_metrics = {
                "keyword_overlap": 0.0,
                "relevance_score": 0.0,
                "retrieval_success": 0.0,
                "avg_processing_time": 0.0,
                "avg_sources_count": 0.0
            }
        
        # Cleanup
        try:
            await retriever.close()
            await llm_cascade.close()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        
        evaluation_summary = {
            "timestamp": time.time(),
            "total_queries": len(self.eval_dataset),
            "successful_queries": len(successful_results),
            "failed_queries": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0.0,
            "aggregate_metrics": avg_metrics,
            "individual_results": results,
            "settings": {
                "llm_provider": self.settings.llm_provider,
                "embedder_name": self.settings.embedder_name,
                "enable_sparse_retrieval": self.settings.enable_sparse_retrieval,
                "enable_reranker": self.settings.enable_reranker,
                "default_top_k": self.settings.default_top_k
            }
        }
        
        logger.info(
            f"Evaluation complete: {len(successful_results)}/{len(results)} successful, "
            f"avg keyword overlap: {avg_metrics['keyword_overlap']:.3f}"
        )
        
        return evaluation_summary


async def run_simple_evaluation(settings: Optional[Settings] = None) -> Dict:
    """Run simple RAG evaluation.
    
    Args:
        settings: Optional settings override
        
    Returns:
        Evaluation results
    """
    if settings is None:
        settings = Settings(
            use_mock_embeddings=True,
            use_mock_llm=True,
            enable_sparse_retrieval=True,
            enable_reranker=False,
            default_top_k=3
        )
    
    evaluator = SimpleRAGEvaluator(settings)
    return await evaluator.run_evaluation()


def print_evaluation_results(results: Dict) -> None:
    """Print evaluation results in a readable format.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("🔍 MEULEX RAG EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Summary:")
    print(f"   Total Queries: {results['total_queries']}")
    print(f"   Successful: {results['successful_queries']}")
    print(f"   Failed: {results['failed_queries']}")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    
    metrics = results['aggregate_metrics']
    print(f"\n📈 Average Metrics:")
    print(f"   Keyword Overlap: {metrics['keyword_overlap']:.3f}")
    print(f"   Relevance Score: {metrics['relevance_score']:.3f}")
    print(f"   Retrieval Success: {metrics['retrieval_success']:.3f}")
    print(f"   Processing Time: {metrics['avg_processing_time']:.3f}s")
    print(f"   Sources per Query: {metrics['avg_sources_count']:.1f}")
    
    print(f"\n⚙️  Configuration:")
    settings = results['settings']
    print(f"   LLM Provider: {settings['llm_provider']}")
    print(f"   Embedder: {settings['embedder_name']}")
    print(f"   Sparse Retrieval: {settings['enable_sparse_retrieval']}")
    print(f"   Reranker: {settings['enable_reranker']}")
    print(f"   Top-K: {settings['default_top_k']}")
    
    print(f"\n📝 Individual Results:")
    for i, result in enumerate(results['individual_results'], 1):
        status = "✅" if result['success'] else "❌"
        print(f"   {i}. {status} {result['question']}")
        if result['success']:
            print(f"      Keywords: {result['metrics']['keyword_overlap']:.3f}, "
                  f"Relevance: {result['metrics']['relevance_score']:.3f}, "
                  f"Sources: {result['sources_count']}")
        else:
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
