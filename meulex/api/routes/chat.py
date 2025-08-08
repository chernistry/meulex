"""Chat endpoint for RAG queries."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator

from meulex.config.settings import get_settings
from meulex.core.caching.cache_manager import CacheManager
from meulex.core.embeddings.factory import create_embedder
from meulex.core.retrieval.hybrid import HybridRetriever
from meulex.core.vector.base import Document
from meulex.core.vector.qdrant_store import QdrantStore
from meulex.llm.base import ChatMessage as LLMChatMessage, LLMMode, MessageRole
from meulex.llm.cascade import LLMCascade
from meulex.llm.prompt_builder import RAGPromptBuilder
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
llm_cascade = None
prompt_builder = None
cache_manager = None


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
    mode: Optional[str] = Field(
        None,
        description="Generation mode (balanced, creative, precise)"
    )
    
    @validator("question")
    def validate_question(cls, v: str) -> str:
        """Validate question field."""
        validate_input_length(v, min_length=1, max_length=2000, field_name="question")
        return sanitize_html(v)
    
    @validator("mode")
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        """Validate mode field."""
        if v is not None:
            valid_modes = ["balanced", "creative", "precise"]
            if v.lower() not in valid_modes:
                raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
            return v.lower()
        return v


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
    global embedder, vector_store, retriever, llm_cascade, prompt_builder, cache_manager
    
    if retriever is None:
        settings = get_settings()
        
        # Initialize cache manager
        cache_manager = CacheManager(settings)
        
        # Initialize embedder and vector store
        embedder = create_embedder(settings)
        vector_store = QdrantStore(settings)
        await vector_store.create_collection()
        
        # Initialize hybrid retriever
        retriever = HybridRetriever(embedder, vector_store, settings)
        
        # Load existing documents into sparse index if enabled
        if settings.enable_sparse_retrieval:
            try:
                # Get all documents from vector store for sparse indexing
                # This is a simplified approach - in production, you'd want a more efficient method
                collection_info = await vector_store.get_collection_info()
                if collection_info.get("points_count", 0) > 0:
                    # For now, we'll skip pre-loading documents into sparse index
                    # In a full implementation, you'd retrieve all documents and add them
                    logger.info("Sparse retrieval enabled but document pre-loading skipped")
            except Exception as e:
                logger.warning(f"Failed to pre-load documents for sparse retrieval: {e}")
        
        # Initialize LLM cascade
        llm_cascade = LLMCascade(settings)
        
        # Initialize prompt builder
        prompt_builder = RAGPromptBuilder(max_context_tokens=settings.max_tokens // 2)
        
        logger.info("Chat endpoint components initialized with hybrid retrieval, LLM cascade, and caching")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """Process a chat query using RAG with LLM cascade.
    
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
            settings = get_settings()
            top_k = request.top_k or settings.default_top_k
            temperature = request.temperature or settings.temperature
            mode = LLMMode(request.mode or "balanced")
            
            span.set_attribute("top_k", top_k)
            span.set_attribute("temperature", temperature)
            span.set_attribute("mode", mode.value)
            
            # Check cache first
            cache_key = None
            cached_response = None
            
            if cache_manager and cache_manager.enabled:
                cache_key = cache_manager.generate_semantic_cache_key(
                    question=request.question,
                    top_k=top_k,
                    mode=mode.value,
                    provider=settings.llm_provider
                )
                
                cached_response = await cache_manager.get(cache_key)
                
                if cached_response:
                    # Return cached response
                    processing_time = time.time() - start_time
                    cached_response["metadata"]["processing_time"] = round(processing_time, 3)
                    cached_response["metadata"]["cached"] = True
                    
                    logger.info(
                        f"Returned cached response for query",
                        extra={
                            "request_id": request_id,
                            "cache_key": cache_key,
                            "processing_time": processing_time
                        }
                    )
                    
                    span.set_attribute("cached", True)
                    span.set_attribute("processing_time", processing_time)
                    
                    return ChatResponse(**cached_response)
            
            span.set_attribute("cached", False)
            
            # Retrieve relevant documents
            documents = await retriever.retrieve(
                query=request.question,
                top_k=top_k
            )
            
            span.set_attribute("documents_retrieved", len(documents))
            
            # Convert chat history to LLM format
            llm_history = []
            for msg in request.history:
                try:
                    role = MessageRole(msg.role.lower())
                    llm_history.append(LLMChatMessage(role=role, content=msg.content))
                except ValueError:
                    # Skip invalid roles
                    continue
            
            # Build prompt with retrieved context
            messages = prompt_builder.build_rag_prompt(
                question=request.question,
                documents=documents,
                conversation_history=llm_history,
                mode=mode
            )
            
            # Generate response using LLM cascade
            llm_response = await llm_cascade.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=settings.max_tokens,
                stream=False  # For now, disable streaming
            )
            
            # Extract answer
            answer = llm_response.content
            
            # Prepare source documents
            sources = []
            for doc in documents:
                # Truncate content for response
                content = doc.content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                source_doc = SourceDocument(
                    text=content,
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
                "model_used": llm_response.model,
                "retrieval_stats": {
                    "documents_found": len(documents),
                    "top_k_requested": top_k,
                    "retrieval_method": "hybrid" if settings.enable_hybrid_retrieval else "dense"
                },
                "llm_stats": {
                    "usage": llm_response.usage.dict() if llm_response.usage else None,
                    "finish_reason": llm_response.finish_reason,
                    "estimated_cost_cents": llm_response.metadata.get("estimated_cost_cents", 0.0)
                },
                "success": True
            }
            
            response = ChatResponse(
                answer=answer,
                sources=sources,
                metadata=metadata
            )
            
            # Cache the response
            if cache_manager and cache_manager.enabled and cache_key:
                try:
                    await cache_manager.set(
                        cache_key,
                        response.dict(),
                        ttl=cache_manager.semantic_ttl
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            # Record metrics
            RAG_REQUESTS.labels(
                status="success",
                provider=llm_response.metadata.get("provider", "unknown")
            ).inc()
            
            RAG_REQUEST_DURATION.observe(processing_time)
            
            logger.info(
                f"Chat query processed: {len(documents)} docs retrieved, {llm_response.usage.total_tokens if llm_response.usage else 0} tokens",
                extra={
                    "request_id": request_id,
                    "question_length": len(request.question),
                    "documents_found": len(documents),
                    "processing_time": processing_time,
                    "model_used": llm_response.model,
                    "total_tokens": llm_response.usage.total_tokens if llm_response.usage else 0
                }
            )
            
            span.set_attribute("success", True)
            span.set_attribute("documents_found", len(documents))
            span.set_attribute("processing_time", processing_time)
            span.set_attribute("model_used", llm_response.model)
            
            return response
            
        except MeulexException:
            # Record error metrics
            RAG_REQUESTS.labels(
                status="error",
                provider="unknown"
            ).inc()
            raise
        except Exception as e:
            logger.error(f"Chat endpoint error: {e}", extra={"request_id": request_id})
            span.set_attribute("error", str(e))
            
            # Record error metrics
            RAG_REQUESTS.labels(
                status="error",
                provider="unknown"
            ).inc()
            
            raise HTTPException(status_code=500, detail="Internal server error")


async def cleanup_chat_components():
    """Cleanup chat endpoint components."""
    global embedder, vector_store, retriever, llm_cascade, prompt_builder, cache_manager
    
    try:
        if cache_manager:
            await cache_manager.close()
        if llm_cascade:
            await llm_cascade.close()
        if retriever:
            await retriever.close()
        if vector_store:
            await vector_store.close()
        if embedder:
            await embedder.close()
        
        logger.info("Chat endpoint components cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up chat components: {e}")
