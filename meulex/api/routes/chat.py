"""Chat endpoint for RAG queries."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator

from meulex.config.settings import get_settings
from meulex.core.embeddings.factory import create_embedder
from meulex.core.retrieval.dense import DenseRetriever
from meulex.core.vector.base import Document
from meulex.core.vector.qdrant_store import QdrantStore
from meulex.observability import RAG_REQUESTS, RAG_REQUEST_DURATION, get_tracer
from meulex.utils.exceptions import MeulexException
from meulex.utils.security import sanitize_html, validate_input_length

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
router = APIRouter()

# Global components (will be initialized on startup)
embedder = None
vector_store = None
retriever = None


class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: str = Field(description="Message role (user, assistant, system)")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    question: str = Field(..., description="User question")
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Conversation history"
    )
    top_k: Optional[int] = Field(
        None,
        description="Number of documents to retrieve",
        ge=1,
        le=20
    )
    temperature: Optional[float] = Field(
        None,
        description="LLM temperature",
        ge=0.0,
        le=2.0
    )
    
    @validator("question")
    def validate_question(cls, v: str) -> str:
        """Validate question field."""
        validate_input_length(v, min_length=1, max_length=2000, field_name="question")
        return sanitize_html(v)


class SourceDocument(BaseModel):
    """Source document model."""
    
    text: str = Field(description="Document text excerpt")
    source: str = Field(description="Document source")
    score: float = Field(description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Document metadata"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    answer: str = Field(description="Generated answer")
    sources: List[SourceDocument] = Field(description="Source documents")
    metadata: Dict[str, Any] = Field(description="Response metadata")


async def initialize_chat_components():
    """Initialize components for chat endpoint."""
    global embedder, vector_store, retriever
    
    if retriever is None:
        settings = get_settings()
        embedder = create_embedder(settings)
        vector_store = QdrantStore(settings)
        await vector_store.create_collection()
        retriever = DenseRetriever(embedder, vector_store, settings)
        
        logger.info("Chat endpoint components initialized")


def generate_simple_answer(query: str, documents: List[Document]) -> str:
    """Generate a simple answer from retrieved documents.
    
    This is a placeholder implementation. In a full system, this would
    use an LLM to generate a proper answer.
    
    Args:
        query: User query
        documents: Retrieved documents
        
    Returns:
        Generated answer
    """
    if not documents:
        return "I couldn't find any relevant information to answer your question."
    
    # Simple answer generation based on document content
    answer_parts = [
        f"Based on the available information, here's what I found about '{query}':",
        ""
    ]
    
    for i, doc in enumerate(documents[:3], 1):
        # Take first sentence or first 200 characters
        content = doc.content.strip()
        if '.' in content:
            first_sentence = content.split('.')[0] + '.'
        else:
            first_sentence = content[:200] + "..." if len(content) > 200 else content
        
        answer_parts.append(f"{i}. {first_sentence}")
    
    if len(documents) > 3:
        answer_parts.append(f"\n(Found {len(documents)} total relevant documents)")
    
    return "\n".join(answer_parts)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """Process a chat query using RAG.
    
    Args:
        request: Chat request
        req: FastAPI request object
        
    Returns:
        Chat response with answer and sources
    """
    start_time = time.time()
    
    with tracer.start_as_current_span("chat_query") as span:
        request_id = getattr(req.state, "request_id", "unknown")
        span.set_attribute("request_id", request_id)
        span.set_attribute("question_length", len(request.question))
        span.set_attribute("history_length", len(request.history))
        
        try:
            # Initialize components if needed
            await initialize_chat_components()
            
            # Set defaults
            top_k = request.top_k or get_settings().default_top_k
            temperature = request.temperature or get_settings().temperature
            
            # Retrieve relevant documents
            documents = await retriever.retrieve(
                query=request.question,
                top_k=top_k
            )
            
            # Generate answer (placeholder implementation)
            answer = generate_simple_answer(request.question, documents)
            
            # Prepare source documents
            sources = []
            for doc in documents:
                source_doc = SourceDocument(
                    text=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    source=doc.metadata.get("source", "unknown"),
                    score=doc.score or 0.0,
                    metadata=doc.metadata
                )
                sources.append(source_doc)
            
            # Prepare metadata
            processing_time = time.time() - start_time
            metadata = {
                "query_id": request_id,
                "processing_time": round(processing_time, 3),
                "model_used": "simple-rag",  # Placeholder
                "retrieval_stats": {
                    "documents_found": len(documents),
                    "top_k_requested": top_k,
                    "reranked": False
                },
                "success": True
            }
            
            response = ChatResponse(
                answer=answer,
                sources=sources,
                metadata=metadata
            )
            
            # Record metrics
            RAG_REQUESTS.labels(
                status="success",
                provider="simple-rag"
            ).inc()
            
            RAG_REQUEST_DURATION.observe(processing_time)
            
            logger.info(
                f"Chat query processed: {len(documents)} docs retrieved",
                extra={
                    "request_id": request_id,
                    "question_length": len(request.question),
                    "documents_found": len(documents),
                    "processing_time": processing_time
                }
            )
            
            span.set_attribute("success", True)
            span.set_attribute("documents_found", len(documents))
            span.set_attribute("processing_time", processing_time)
            
            return response
            
        except MeulexException:
            # Record error metrics
            RAG_REQUESTS.labels(
                status="error",
                provider="simple-rag"
            ).inc()
            raise
        except Exception as e:
            logger.error(f"Chat endpoint error: {e}", extra={"request_id": request_id})
            span.set_attribute("error", str(e))
            
            # Record error metrics
            RAG_REQUESTS.labels(
                status="error",
                provider="simple-rag"
            ).inc()
            
            raise HTTPException(status_code=500, detail="Internal server error")


async def cleanup_chat_components():
    """Cleanup chat endpoint components."""
    global embedder, vector_store, retriever
    
    try:
        if retriever:
            await retriever.close()
        if vector_store:
            await vector_store.close()
        if embedder:
            await embedder.close()
        
        logger.info("Chat endpoint components cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up chat components: {e}")
