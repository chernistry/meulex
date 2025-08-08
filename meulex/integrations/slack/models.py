"""Slack event models and types."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SlackEventType(str, Enum):
    """Slack event types."""
    
    URL_VERIFICATION = "url_verification"
    EVENT_CALLBACK = "event_callback"
    APP_MENTION = "app_mention"
    MESSAGE = "message"


class SlackUser(BaseModel):
    """Slack user model."""
    
    id: str = Field(description="User ID")
    name: Optional[str] = Field(None, description="User name")
    real_name: Optional[str] = Field(None, description="Real name")


class SlackChannel(BaseModel):
    """Slack channel model."""
    
    id: str = Field(description="Channel ID")
    name: Optional[str] = Field(None, description="Channel name")


class SlackMessage(BaseModel):
    """Slack message model."""
    
    type: str = Field(description="Message type")
    subtype: Optional[str] = Field(None, description="Message subtype")
    text: str = Field(description="Message text")
    user: str = Field(description="User ID")
    ts: str = Field(description="Message timestamp")
    channel: str = Field(description="Channel ID")
    thread_ts: Optional[str] = Field(None, description="Thread timestamp")
    bot_id: Optional[str] = Field(None, description="Bot ID if from bot")


class SlackEvent(BaseModel):
    """Slack event model."""
    
    type: str = Field(description="Event type")
    user: Optional[str] = Field(None, description="User ID")
    text: Optional[str] = Field(None, description="Event text")
    ts: Optional[str] = Field(None, description="Event timestamp")
    channel: Optional[str] = Field(None, description="Channel ID")
    thread_ts: Optional[str] = Field(None, description="Thread timestamp")
    event_ts: Optional[str] = Field(None, description="Event timestamp")


class SlackEventPayload(BaseModel):
    """Slack Events API payload."""
    
    token: str = Field(description="Verification token")
    team_id: Optional[str] = Field(None, description="Team ID")
    api_app_id: Optional[str] = Field(None, description="App ID")
    event: Optional[SlackEvent] = Field(None, description="Event data")
    type: str = Field(description="Payload type")
    event_id: Optional[str] = Field(None, description="Event ID")
    event_time: Optional[int] = Field(None, description="Event time")
    challenge: Optional[str] = Field(None, description="URL verification challenge")


class SlackResponse(BaseModel):
    """Slack response model."""
    
    text: str = Field(description="Response text")
    response_type: str = Field(default="ephemeral", description="Response type")
    thread_ts: Optional[str] = Field(None, description="Thread timestamp for replies")
    blocks: Optional[List[Dict[str, Any]]] = Field(None, description="Block kit blocks")


class SlackMessageFormat:
    """Slack message formatting utilities."""
    
    @staticmethod
    def format_rag_response(
        answer: str,
        sources: List[Dict[str, Any]],
        query: str,
        processing_time: float
    ) -> SlackResponse:
        """Format RAG response for Slack.
        
        Args:
            answer: Generated answer
            sources: Source documents
            query: Original query
            processing_time: Processing time in seconds
            
        Returns:
            Formatted Slack response
        """
        # Truncate answer if too long (Slack has message limits)
        max_answer_length = 2000
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "... [truncated]"
        
        # Format sources
        source_text = ""
        if sources:
            source_text = "\n\n*Sources:*\n"
            for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources
                source_name = source.get("source", "Unknown")
                score = source.get("score", 0.0)
                source_text += f"• {source_name} (relevance: {score:.2f})\n"
        
        # Add footer with processing time
        footer = f"\n_Processed in {processing_time:.2f}s_"
        
        # Combine all parts
        full_text = f"{answer}{source_text}{footer}"
        
        return SlackResponse(
            text=full_text,
            response_type="ephemeral"  # Only visible to the user who asked
        )
    
    @staticmethod
    def format_error_response(error_message: str) -> SlackResponse:
        """Format error response for Slack.
        
        Args:
            error_message: Error message
            
        Returns:
            Formatted error response
        """
        return SlackResponse(
            text=f"❌ Sorry, I encountered an error: {error_message}",
            response_type="ephemeral"
        )
    
    @staticmethod
    def format_help_response() -> SlackResponse:
        """Format help response for Slack.
        
        Returns:
            Formatted help response
        """
        help_text = """
🤖 *Meulex AI Assistant*

I can help you find information from our knowledge base. Here's how to use me:

*Commands:*
• `@meulex <your question>` - Ask me anything about our documentation
• `@meulex help` - Show this help message

*Examples:*
• `@meulex How do I deploy the application?`
• `@meulex What are the API endpoints?`
• `@meulex Tell me about the security features`

I'll search through our documents and provide you with relevant information along with source citations.
        """.strip()
        
        return SlackResponse(
            text=help_text,
            response_type="ephemeral"
        )
