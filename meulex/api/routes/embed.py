"""Embed endpoint for document ingestion."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator

from meulex.config.settings import get_settings
from meulex.core.embeddings.factory import create_embedder
from meulex.core.ingestion.ingestor import DocumentIngestor
from meulex.core.vector.qdrant_store import QdrantStore
from meulex.observability import EMBEDDINGS_GENERATED, get_tracer
from meulex.utils.exceptions import MeulexException, ValidationError
from meulex.utils.security import sanitize_html, validate_input_length

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
router = APIRouter()

# Global components (will be initialized on startup)
embedder = None
vector_store = None
ingestor = None


class EmbedRequest(BaseModel):
    """Request model for embed endpoint."""
    
    id: Optional[str] = Field(None, description="Optional document ID")
    content: str = Field(..., description="Content to embed")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Optional metadata"
    )
    
    @validator("content")
    def validate_content(cls, v: str) -> str:
        """Validate content field."""
        validate_input_length(v, min_length=1, max_length=50000, field_name="content")
        return sanitize_html(v)
    
    @validator("id")
    def validate_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate ID field."""
        if v is not None:
            validate_input_length(v, min_length=1, max_length=100, field_name="id")
        return v


class EmbedResponse(BaseModel):
    """Response model for embed endpoint."""
    
    status: str = Field(description="Status of the operation")
    count: int = Field(description="Number of chunks created")
    ids: list[str] = Field(description="List of chunk IDs")
    document_id: Optional[str] = Field(None, description="Document ID")


async def initialize_embed_components():
    """Initialize components for embed endpoint."""
    global embedder, vector_store, ingestor
    
    if embedder is None:
        settings = get_settings()
        embedder = create_embedder(settings)
        vector_store = QdrantStore(settings)
        await vector_store.create_collection()
        ingestor = DocumentIngestor(embedder, vector_store, settings)
        
        logger.info("Embed endpoint components initialized")


@router.post("/embed", response_model=EmbedResponse)
async def embed_document(request: EmbedRequest, req: Request) -> EmbedResponse:
    """Embed a document and store it in the vector database.
    
    Args:
        request: Embed request
        req: FastAPI request object
        
    Returns:
        Embed response with chunk information
    """
    with tracer.start_as_current_span("embed_document") as span:
        request_id = getattr(req.state, "request_id", "unknown")
        span.set_attribute("request_id", request_id)
        span.set_attribute("content_length", len(request.content))
        span.set_attribute("has_metadata", bool(request.metadata))
        
        try:
            # Initialize components if needed
            await initialize_embed_components()
            
            # Generate source identifier
            source = request.id or f"api_request_{request_id}"
            
            # Ingest document
            result = await ingestor.ingest_document(
                content=request.content,
                source=source,
                metadata=request.metadata
            )
            
            if result["status"] != "success":
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to embed document: {result.get('error', 'Unknown error')}"
                )
            
            # Record metrics
            EMBEDDINGS_GENERATED.labels(
                model=embedder.model_name,
                provider="api"
            ).inc(result["chunk_count"])
            
            response = EmbedResponse(
                status="ok",
                count=result["chunk_count"],
                ids=result["chunk_ids"],
                document_id=result["document_id"]
            )
            
            logger.info(
                f"Embedded document via API: {result['chunk_count']} chunks",
                extra={
                    "request_id": request_id,
                    "document_id": result["document_id"],
                    "chunk_count": result["chunk_count"]
                }
            )
            
            span.set_attribute("success", True)
            span.set_attribute("chunk_count", result["chunk_count"])
            
            return response
            
        except MeulexException:
            raise
        except Exception as e:
            logger.error(f"Embed endpoint error: {e}", extra={"request_id": request_id})
            span.set_attribute("error", str(e))
            raise HTTPException(status_code=500, detail="Internal server error")


async def cleanup_embed_components():
    """Cleanup embed endpoint components."""
    global embedder, vector_store, ingestor
    
    try:
        if ingestor:
            await ingestor.close()
        if vector_store:
            await vector_store.close()
        if embedder:
            await embedder.close()
        
        logger.info("Embed endpoint components cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up embed components: {e}")
