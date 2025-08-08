"""Base LLM interface and models."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message roles for chat completion."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: MessageRole = Field(description="Message role")
    content: str = Field(description="Message content")


class LLMMode(str, Enum):
    """LLM generation modes."""
    
    BALANCED = "balanced"
    CREATIVE = "creative"
    PRECISE = "precise"


class LLMUsage(BaseModel):
    """LLM usage statistics."""
    
    prompt_tokens: int = Field(description="Number of prompt tokens")
    completion_tokens: int = Field(description="Number of completion tokens")
    total_tokens: int = Field(description="Total number of tokens")


class LLMResponse(BaseModel):
    """LLM response model."""
    
    content: str = Field(description="Generated content")
    usage: Optional[LLMUsage] = Field(None, description="Token usage statistics")
    model: str = Field(description="Model used for generation")
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate chat completion.
        
        Args:
            messages: List of chat messages
            model: Model to use (optional override)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM response or async generator for streaming
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the provider and cleanup resources."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model name."""
        pass
    
    @abstractmethod
    def estimate_cost(self, usage: LLMUsage, model: str) -> float:
        """Estimate cost in cents for the given usage.
        
        Args:
            usage: Token usage statistics
            model: Model name
            
        Returns:
            Estimated cost in cents
        """
        pass
