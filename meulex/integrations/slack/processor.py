"""Slack event processor."""

import asyncio
import logging
import re
import time
from typing import Optional

import httpx

from meulex.config.settings import Settings
from meulex.core.caching.cache_manager import CacheManager
from meulex.integrations.slack.models import (
    SlackEvent,
    SlackEventPayload,
    SlackMessageFormat,
    SlackResponse,
)
from meulex.observability import SLACK_EVENTS, get_tracer
from meulex.utils.exceptions import SlackError

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class SlackEventProcessor:
    """Processes Slack events and generates responses."""
    
    def __init__(
        self,
        settings: Settings,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize Slack event processor.
        
        Args:
            settings: Application settings
            cache_manager: Optional cache manager for idempotency
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.bot_user_id = settings.slack_bot_user_id
        
        # HTTP client for Slack API
        self.slack_client = httpx.AsyncClient(
            base_url="https://slack.com/api",
            headers={
                "Authorization": f"Bearer {settings.slack_bot_token}",
                "Content-Type": "application/json"
            },
            timeout=settings.request_timeout_s
        ) if settings.slack_bot_token else None
        
        logger.info(
            "Slack event processor initialized",
            extra={
                "has_bot_token": bool(settings.slack_bot_token),
                "bot_user_id": self.bot_user_id
            }
        )
    
    async def process_event(
        self,
        payload: SlackEventPayload,
        chat_handler: callable
    ) -> Optional[SlackResponse]:
        """Process a Slack event.
        
        Args:
            payload: Slack event payload
            chat_handler: Function to handle chat queries
            
        Returns:
            Slack response or None
        """
        with tracer.start_as_current_span("slack_process_event") as span:
            span.set_attribute("event_type", payload.type)
            span.set_attribute("event_id", payload.event_id or "unknown")
            
            try:
                # Handle URL verification
                if payload.type == "url_verification":
                    logger.info("Handling URL verification challenge")
                    span.set_attribute("challenge", bool(payload.challenge))
                    return None  # Challenge is handled by the endpoint directly
                
                # Handle event callbacks
                if payload.type == "event_callback" and payload.event:
                    return await self._process_event_callback(
                        payload.event,
                        payload.event_id,
                        chat_handler,
                        span
                    )
                
                logger.warning(f"Unhandled event type: {payload.type}")
                span.set_attribute("unhandled", True)
                return None
                
            except Exception as e:
                logger.error(f"Error processing Slack event: {e}")
                span.set_attribute("error", str(e))
                
                # Record error metrics
                SLACK_EVENTS.labels(
                    event_type=payload.type,
                    status="error"
                ).inc()
                
                return SlackMessageFormat.format_error_response(
                    "I encountered an error processing your request. Please try again."
                )
    
    async def _process_event_callback(
        self,
        event: SlackEvent,
        event_id: Optional[str],
        chat_handler: callable,
        span
    ) -> Optional[SlackResponse]:
        """Process event callback.
        
        Args:
            event: Slack event
            event_id: Event ID for idempotency
            chat_handler: Function to handle chat queries
            span: OpenTelemetry span
            
        Returns:
            Slack response or None
        """
        span.set_attribute("event_callback_type", event.type)
        
        # Check for idempotency
        if event_id and self.cache_manager:
            idempotency_key = f"slack_event:{event_id}"
            cached_response = await self.cache_manager.get(idempotency_key)
            
            if cached_response:
                logger.info(f"Returning cached response for event {event_id}")
                span.set_attribute("idempotent", True)
                return SlackResponse(**cached_response)
        
        # Handle app mentions
        if event.type == "app_mention":
            response = await self._handle_app_mention(event, chat_handler, span)
        
        # Handle direct messages
        elif event.type == "message" and event.channel and event.channel.startswith("D"):
            response = await self._handle_direct_message(event, chat_handler, span)
        
        else:
            logger.debug(f"Ignoring event type: {event.type}")
            return None
        
        # Cache response for idempotency
        if response and event_id and self.cache_manager:
            try:
                await self.cache_manager.set(
                    idempotency_key,
                    response.dict(),
                    ttl=900  # 15 minutes
                )
            except Exception as e:
                logger.warning(f"Failed to cache Slack response: {e}")
        
        return response
    
    async def _handle_app_mention(
        self,
        event: SlackEvent,
        chat_handler: callable,
        span
    ) -> Optional[SlackResponse]:
        """Handle app mention events.
        
        Args:
            event: Slack event
            chat_handler: Function to handle chat queries
            span: OpenTelemetry span
            
        Returns:
            Slack response or None
        """
        if not event.text or not event.user:
            return None
        
        # Skip bot messages and messages from the bot itself
        if event.user == self.bot_user_id or event.bot_id:
            logger.debug(f"Ignoring bot message from user {event.user} (bot_user_id: {self.bot_user_id}, bot_id: {event.bot_id})")
            return None
        
        # Skip message subtypes that shouldn't trigger responses (like edits, deletes, etc.)
        if hasattr(event, 'subtype') and event.subtype:
            logger.debug(f"Ignoring message with subtype: {event.subtype}")
            return None
        
        span.set_attribute("user_id", event.user)
        span.set_attribute("channel_id", event.channel or "unknown")
        
        # Extract question from mention
        question = self._extract_question_from_mention(event.text)
        
        if not question:
            return SlackMessageFormat.format_help_response()
        
        # Handle help requests
        if question.lower().strip() in ["help", "?"]:
            return SlackMessageFormat.format_help_response()
        
        span.set_attribute("question_length", len(question))
        
        # Process the question
        try:
            start_time = time.time()
            
            # Call the chat handler
            chat_response = await chat_handler(question)
            
            processing_time = time.time() - start_time
            
            # Format response for Slack
            response = SlackMessageFormat.format_rag_response(
                answer=chat_response["answer"],
                sources=chat_response["sources"],
                query=question,
                processing_time=processing_time
            )
            
            # Set thread_ts for threaded replies
            if event.thread_ts:
                response.thread_ts = event.thread_ts
            elif event.ts:
                response.thread_ts = event.ts
            
            # Record success metrics
            SLACK_EVENTS.labels(
                event_type="app_mention",
                status="success"
            ).inc()
            
            logger.info(
                f"Processed app mention: {len(question)} chars, {processing_time:.2f}s",
                extra={
                    "user_id": event.user,
                    "channel_id": event.channel,
                    "question_length": len(question),
                    "processing_time": processing_time
                }
            )
            
            span.set_attribute("success", True)
            span.set_attribute("processing_time", processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing app mention: {e}")
            span.set_attribute("error", str(e))
            
            # Record error metrics
            SLACK_EVENTS.labels(
                event_type="app_mention",
                status="error"
            ).inc()
            
            return SlackMessageFormat.format_error_response(
                "I encountered an error processing your question. Please try again."
            )
    
    async def _handle_direct_message(
        self,
        event: SlackEvent,
        chat_handler: callable,
        span
    ) -> Optional[SlackResponse]:
        """Handle direct message events.
        
        Args:
            event: Slack event
            chat_handler: Function to handle chat queries
            span: OpenTelemetry span
            
        Returns:
            Slack response or None
        """
        if not event.text or not event.user:
            return None
        
        # Skip bot messages and messages from the bot itself
        if event.user == self.bot_user_id or event.bot_id:
            return None
        
        # Skip message subtypes that shouldn't trigger responses
        if hasattr(event, 'subtype') and event.subtype:
            return None
        
        span.set_attribute("user_id", event.user)
        span.set_attribute("is_dm", True)
        
        question = event.text.strip()
        
        # Handle help requests
        if question.lower() in ["help", "?"]:
            return SlackMessageFormat.format_help_response()
        
        # Process similar to app mention but without mention extraction
        try:
            start_time = time.time()
            
            chat_response = await chat_handler(question)
            processing_time = time.time() - start_time
            
            response = SlackMessageFormat.format_rag_response(
                answer=chat_response["answer"],
                sources=chat_response["sources"],
                query=question,
                processing_time=processing_time
            )
            
            # Record success metrics
            SLACK_EVENTS.labels(
                event_type="direct_message",
                status="success"
            ).inc()
            
            span.set_attribute("success", True)
            return response
            
        except Exception as e:
            logger.error(f"Error processing direct message: {e}")
            span.set_attribute("error", str(e))
            
            # Record error metrics
            SLACK_EVENTS.labels(
                event_type="direct_message",
                status="error"
            ).inc()
            
            return SlackMessageFormat.format_error_response(
                "I encountered an error processing your message. Please try again."
            )
    
    def _extract_question_from_mention(self, text: str) -> str:
        """Extract question from app mention text.
        
        Args:
            text: Raw mention text
            
        Returns:
            Extracted question
        """
        # Remove bot mention (e.g., "<@U1234567890>")
        mention_pattern = r'<@[UW][A-Z0-9]+>'
        text = re.sub(mention_pattern, '', text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def send_response(
        self,
        response: SlackResponse,
        channel: str,
        thread_ts: Optional[str] = None
    ) -> bool:
        """Send response to Slack channel with idempotency protection.
        
        Args:
            response: Slack response
            channel: Channel ID
            thread_ts: Optional thread timestamp
            
        Returns:
            True if successful
        """
        if not self.slack_client:
            logger.warning("Slack client not configured - cannot send response")
            return False
        
        with tracer.start_as_current_span("slack_send_response") as span:
            span.set_attribute("channel", channel)
            span.set_attribute("has_thread", bool(thread_ts))
            
            # Create idempotency key for message sending
            message_key = f"slack_message:{channel}:{thread_ts or 'no_thread'}:{hash(response.text)}"
            
            # Check if we've already sent this exact message
            if self.cache_manager:
                already_sent = await self.cache_manager.get(message_key)
                if already_sent:
                    logger.info(f"Message already sent to {channel}, skipping duplicate")
                    span.set_attribute("duplicate_prevented", True)
                    return True
            
            try:
                payload = {
                    "channel": channel,
                    "text": response.text,
                    "response_type": response.response_type
                }
                
                if thread_ts or response.thread_ts:
                    payload["thread_ts"] = thread_ts or response.thread_ts
                
                if response.blocks:
                    payload["blocks"] = response.blocks
                
                slack_response = await self.slack_client.post(
                    "/chat.postMessage",
                    json=payload
                )
                
                slack_response.raise_for_status()
                result = slack_response.json()
                
                if result.get("ok"):
                    logger.info(f"Sent Slack response to {channel}")
                    span.set_attribute("success", True)
                    
                    # Cache that we sent this message to prevent duplicates
                    if self.cache_manager:
                        try:
                            await self.cache_manager.set(
                                message_key,
                                {"sent_at": time.time()},
                                ttl=300  # 5 minutes
                            )
                        except Exception as e:
                            logger.warning(f"Failed to cache message send status: {e}")
                    
                    return True
                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"Slack API error: {error}", extra={
                        "channel": channel,
                        "slack_error": error,
                        "full_response": result
                    })
                    span.set_attribute("slack_error", error)
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to send Slack response: {e}")
                span.set_attribute("error", str(e))
                return False
    
    async def close(self) -> None:
        """Close the Slack client."""
        if self.slack_client:
            try:
                await self.slack_client.aclose()
                logger.info("Slack client closed")
            except Exception as e:
                logger.warning(f"Error closing Slack client: {e}")
