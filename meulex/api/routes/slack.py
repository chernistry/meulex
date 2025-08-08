"""Slack Events API endpoint."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError

from meulex.config.settings import get_settings
from meulex.core.caching.cache_manager import CacheManager
from meulex.core.embeddings.factory import create_embedder
from meulex.core.retrieval.hybrid import HybridRetriever
from meulex.core.vector.qdrant_store import QdrantStore
from meulex.integrations.slack.auth import SlackSignatureVerifier
from meulex.integrations.slack.models import SlackEventPayload
from meulex.integrations.slack.processor import SlackEventProcessor
from meulex.llm.base import LLMMode
from meulex.llm.cascade import LLMCascade
from meulex.llm.prompt_builder import RAGPromptBuilder
from meulex.observability import SLACK_EVENTS, get_tracer
from meulex.utils.exceptions import AuthenticationError, MeulexException

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
router = APIRouter()

# Global components (will be initialized on startup)
signature_verifier = None
event_processor = None
cache_manager = None
embedder = None
vector_store = None
retriever = None
llm_cascade = None
prompt_builder = None


async def initialize_slack_components():
    """Initialize components for Slack endpoint."""
    global signature_verifier, event_processor, cache_manager
    global embedder, vector_store, retriever, llm_cascade, prompt_builder
    
    if signature_verifier is None:
        settings = get_settings()
        
        # Initialize signature verifier
        signature_verifier = SlackSignatureVerifier(settings)
        
        # Initialize cache manager
        cache_manager = CacheManager(settings)
        
        # Initialize RAG components (reuse from chat endpoint)
        embedder = create_embedder(settings)
        vector_store = QdrantStore(settings)
        await vector_store.create_collection()
        retriever = HybridRetriever(embedder, vector_store, settings)
        llm_cascade = LLMCascade(settings)
        prompt_builder = RAGPromptBuilder(max_context_tokens=settings.max_tokens // 2)
        
        # Initialize event processor
        event_processor = SlackEventProcessor(settings, cache_manager)
        
        logger.info("Slack endpoint components initialized")


async def handle_chat_query(question: str) -> Dict[str, Any]:
    """Handle chat query for Slack.
    
    Args:
        question: User question
        
    Returns:
        Chat response dictionary
    """
    settings = get_settings()
    
    # Retrieve relevant documents
    documents = await retriever.retrieve(
        query=question,
        top_k=settings.default_top_k
    )
    
    # Build prompt
    messages = prompt_builder.build_rag_prompt(
        question=question,
        documents=documents,
        conversation_history=None,  # No history for Slack
        mode=LLMMode.BALANCED
    )
    
    # Generate response
    llm_response = await llm_cascade.chat_completion(
        messages=messages,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stream=False
    )
    
    # Format sources
    sources = []
    for doc in documents:
        sources.append({
            "text": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
            "source": doc.metadata.get("source", "unknown"),
            "score": doc.score or 0.0,
            "metadata": doc.metadata
        })
    
    return {
        "answer": llm_response.content,
        "sources": sources,
        "metadata": {
            "model_used": llm_response.model,
            "documents_found": len(documents),
            "usage": llm_response.usage.dict() if llm_response.usage else None
        }
    }


@router.post("/slack/events")
async def slack_events(request: Request) -> Dict[str, Any]:
    """Handle Slack Events API requests.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Response for Slack
    """
    with tracer.start_as_current_span("slack_events") as span:
        request_id = getattr(request.state, "request_id", "unknown")
        span.set_attribute("request_id", request_id)
        
        try:
            # Initialize components if needed
            await initialize_slack_components()
            
            # Get raw body and headers
            body = await request.body()
            headers = dict(request.headers)
            
            span.set_attribute("body_length", len(body))
            span.set_attribute("has_signature", "x-slack-signature" in headers)
            
            # Extract Slack headers
            timestamp, signature = signature_verifier.extract_headers(headers)
            
            if not timestamp or not signature:
                logger.warning("Missing Slack headers")
                raise HTTPException(
                    status_code=400,
                    detail="Missing required Slack headers"
                )
            
            # Verify signature (can be disabled for debugging)
            settings = get_settings()
            if settings.slack_signature_verification_enabled:
                try:
                    signature_verifier.verify_signature(timestamp, body, signature)
                    logger.debug("Slack signature verification successful")
                except AuthenticationError as e:
                    logger.warning(f"Slack signature verification failed: {e}")
                    raise HTTPException(status_code=401, detail="Invalid signature")
            else:
                logger.warning("Slack signature verification is DISABLED - only use for debugging!")
            
            # Parse payload
            try:
                payload_data = await request.json()
                payload = SlackEventPayload(**payload_data)
            except ValidationError as e:
                logger.error(f"Invalid Slack payload: {e}")
                raise HTTPException(status_code=400, detail="Invalid payload format")
            
            span.set_attribute("event_type", payload.type)
            span.set_attribute("event_id", payload.event_id or "unknown")
            
            # Handle URL verification challenge
            if payload.type == "url_verification":
                if payload.challenge:
                    logger.info("Responding to URL verification challenge")
                    span.set_attribute("challenge_response", True)
                    return {"challenge": payload.challenge}
                else:
                    raise HTTPException(status_code=400, detail="Missing challenge")
            
            # Process event
            response = await event_processor.process_event(payload, handle_chat_query)
            
            # Record metrics
            SLACK_EVENTS.labels(
                event_type=payload.type,
                status="success"
            ).inc()
            
            # For event callbacks, we need to respond quickly (within 3 seconds)
            # and then optionally send a response via the Slack API
            if payload.type == "event_callback":
                logger.debug(f"Processing event_callback: {payload.event.type if payload.event else 'no event'}")
                
                # Send immediate acknowledgment
                ack_response = {"status": "ok"}
                logger.debug("Sending ACK response to Slack")
                
                # If we have a response to send, send it asynchronously
                if response and payload.event:
                    logger.debug(f"Scheduling async response to channel {payload.event.channel}")
                    # Don't await this - send it in the background
                    import asyncio
                    asyncio.create_task(
                        event_processor.send_response(
                            response,
                            payload.event.channel,
                            payload.event.thread_ts
                        )
                    )
                else:
                    logger.debug("No response to send or no event data")
                
                span.set_attribute("success", True)
                return ack_response
            
            # For other event types, return the response directly
            span.set_attribute("success", True)
            return response.dict() if response else {"status": "ok"}
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except MeulexException as e:
            logger.error(f"Meulex error in Slack endpoint: {e}")
            span.set_attribute("error", str(e))
            
            # Record error metrics
            SLACK_EVENTS.labels(
                event_type="unknown",
                status="error"
            ).inc()
            
            raise HTTPException(status_code=500, detail="Internal server error")
        except Exception as e:
            logger.error(f"Unexpected error in Slack endpoint: {e}")
            span.set_attribute("error", str(e))
            
            # Record error metrics
            SLACK_EVENTS.labels(
                event_type="unknown",
                status="error"
            ).inc()
            
            raise HTTPException(status_code=500, detail="Internal server error")


async def cleanup_slack_components():
    """Cleanup Slack endpoint components."""
    global signature_verifier, event_processor, cache_manager
    global embedder, vector_store, retriever, llm_cascade, prompt_builder
    
    try:
        if event_processor:
            await event_processor.close()
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
        
        logger.info("Slack endpoint components cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up Slack components: {e}")
