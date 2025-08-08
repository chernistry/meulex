"""OpenAI-compatible LLM provider."""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from meulex.config.settings import Settings
from meulex.llm.base import BaseLLMProvider, ChatMessage, LLMResponse, LLMUsage
from meulex.observability import LLM_COST, LLM_FAILURES, LLM_TOKENS, get_tracer
from meulex.utils.exceptions import BudgetExceededError, LLMError

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider."""
    
    def __init__(self, settings: Settings) -> None:
        """Initialize OpenAI provider.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.organization = settings.openai_organization
        self.model = settings.llm_model
        self.timeout = settings.request_timeout_s
        
        if not self.api_key:
            raise LLMError(
                "OpenAI API key is required",
                "openai",
                {"model": self.model}
            )
        
        # HTTP client
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout
        )
        
        # Cost tracking (rough estimates in cents per 1K tokens)
        self.cost_per_1k_tokens = {
            "gpt-3.5-turbo": {"prompt": 0.15, "completion": 0.20},
            "gpt-3.5-turbo-16k": {"prompt": 0.30, "completion": 0.40},
            "gpt-4": {"prompt": 3.00, "completion": 6.00},
            "gpt-4-32k": {"prompt": 6.00, "completion": 12.00},
            "gpt-4-turbo": {"prompt": 1.00, "completion": 3.00},
            "gpt-4o": {"prompt": 0.50, "completion": 1.50},
        }
        
        logger.info(
            "OpenAI provider initialized",
            extra={
                "model": self.model,
                "base_url": self.base_url,
                "has_organization": bool(self.organization)
            }
        )
    
    @property
    def name(self) -> str:
        """Get the provider name."""
        return "openai"
    
    @property
    def default_model(self) -> str:
        """Get the default model name."""
        return self.model
    
    def estimate_cost(self, usage: LLMUsage, model: str) -> float:
        """Estimate cost in cents for the given usage.
        
        Args:
            usage: Token usage statistics
            model: Model name
            
        Returns:
            Estimated cost in cents
        """
        if model not in self.cost_per_1k_tokens:
            # Default to GPT-3.5-turbo pricing
            model = "gpt-3.5-turbo"
        
        costs = self.cost_per_1k_tokens[model]
        prompt_cost = (usage.prompt_tokens / 1000) * costs["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * costs["completion"]
        
        return prompt_cost + completion_cost
    
    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
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
        with tracer.start_as_current_span("openai_chat_completion") as span:
            model = model or self.model
            max_tokens = max_tokens or self.settings.max_tokens
            
            span.set_attribute("model", model)
            span.set_attribute("temperature", temperature)
            span.set_attribute("max_tokens", max_tokens)
            span.set_attribute("stream", stream)
            span.set_attribute("message_count", len(messages))
            
            try:
                # Prepare request payload
                payload = {
                    "model": model,
                    "messages": [
                        {"role": msg.role.value, "content": msg.content}
                        for msg in messages
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                    **kwargs
                }
                
                # Make request
                response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                
                if stream:
                    return self._handle_streaming_response(response, model, span)
                else:
                    return await self._handle_response(response, model, span)
                    
            except httpx.HTTPStatusError as e:
                error_msg = f"OpenAI API error: {e.response.status_code}"
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {error_detail.get('error', {}).get('message', 'Unknown error')}"
                except Exception:
                    error_msg += f" - {e.response.text}"
                
                logger.error(error_msg)
                span.set_attribute("error", error_msg)
                
                # Record failure metrics
                LLM_FAILURES.labels(
                    provider="openai",
                    error_type="http_error"
                ).inc()
                
                raise LLMError(
                    error_msg,
                    "openai",
                    {
                        "status_code": e.response.status_code,
                        "model": model,
                        "message_count": len(messages)
                    }
                )
                
            except Exception as e:
                error_msg = f"Failed to generate completion: {e}"
                logger.error(error_msg)
                span.set_attribute("error", error_msg)
                
                # Record failure metrics
                LLM_FAILURES.labels(
                    provider="openai",
                    error_type="unknown"
                ).inc()
                
                raise LLMError(
                    error_msg,
                    "openai",
                    {
                        "model": model,
                        "message_count": len(messages),
                        "error_type": type(e).__name__
                    }
                )
    
    async def _handle_response(
        self,
        response: httpx.Response,
        model: str,
        span
    ) -> LLMResponse:
        """Handle non-streaming response.
        
        Args:
            response: HTTP response
            model: Model name
            span: OpenTelemetry span
            
        Returns:
            LLM response
        """
        data = response.json()
        
        # Extract response data
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        
        # Extract usage statistics
        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        # Estimate cost
        estimated_cost = self.estimate_cost(usage, model)
        
        # Check budget
        if (self.settings.enable_budget_enforcement and 
            estimated_cost > self.settings.cost_budget_cents):
            raise BudgetExceededError(
                f"Request would exceed budget: {estimated_cost:.2f} cents",
                "cost",
                {"estimated_cost": estimated_cost, "budget": self.settings.cost_budget_cents}
            )
        
        # Record metrics
        LLM_TOKENS.labels(
            provider="openai",
            type="prompt"
        ).inc(usage.prompt_tokens)
        
        LLM_TOKENS.labels(
            provider="openai",
            type="completion"
        ).inc(usage.completion_tokens)
        
        LLM_COST.labels(provider="openai").inc(estimated_cost)
        
        # Create response
        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=model,
            finish_reason=finish_reason,
            metadata={
                "estimated_cost_cents": estimated_cost,
                "provider": "openai"
            }
        )
        
        logger.info(
            f"Generated completion: {usage.total_tokens} tokens, ~{estimated_cost:.2f}¢",
            extra={
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "estimated_cost": estimated_cost
            }
        )
        
        span.set_attribute("success", True)
        span.set_attribute("total_tokens", usage.total_tokens)
        span.set_attribute("estimated_cost", estimated_cost)
        
        return llm_response
    
    async def _handle_streaming_response(
        self,
        response: httpx.Response,
        model: str,
        span
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response.
        
        Args:
            response: HTTP response
            model: Model name
            span: OpenTelemetry span
            
        Yields:
            Content chunks
        """
        # Note: This is a simplified streaming implementation
        # In production, you'd want to properly parse SSE events
        
        total_tokens = 0
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                
                if data_str == "[DONE]":
                    break
                
                try:
                    import json
                    data = json.loads(data_str)
                    
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            total_tokens += 1  # Rough approximation
                            yield content
                            
                except json.JSONDecodeError:
                    continue
        
        # Record approximate metrics for streaming
        LLM_TOKENS.labels(
            provider="openai",
            type="completion"
        ).inc(total_tokens)
        
        span.set_attribute("success", True)
        span.set_attribute("streaming", True)
        span.set_attribute("approximate_tokens", total_tokens)
    
    async def close(self) -> None:
        """Close the HTTP client."""
        try:
            await self.client.aclose()
            logger.info("OpenAI provider client closed")
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}")


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for development/testing."""
    
    def __init__(self, settings: Settings) -> None:
        """Initialize mock LLM provider.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = "mock-llm"
        
        logger.info("Mock LLM provider initialized")
    
    @property
    def name(self) -> str:
        """Get the provider name."""
        return "mock"
    
    @property
    def default_model(self) -> str:
        """Get the default model name."""
        return self.model
    
    def estimate_cost(self, usage: LLMUsage, model: str) -> float:
        """Estimate cost (always 0 for mock).
        
        Args:
            usage: Token usage statistics
            model: Model name
            
        Returns:
            Always 0.0 for mock provider
        """
        return 0.0
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate mock chat completion.
        
        Args:
            messages: List of chat messages
            model: Model to use (ignored)
            temperature: Generation temperature (ignored)
            max_tokens: Maximum tokens to generate (ignored)
            stream: Whether to stream the response
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock LLM response or async generator for streaming
        """
        with tracer.start_as_current_span("mock_chat_completion") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("message_count", len(messages))
            span.set_attribute("stream", stream)
            
            # Generate a simple mock response based on the last user message
            last_user_message = None
            for msg in reversed(messages):
                if msg.role.value == "user":
                    last_user_message = msg.content
                    break
            
            if last_user_message:
                mock_content = f"This is a mock response to: '{last_user_message[:50]}...'. In a real implementation, this would be generated by an LLM based on the provided context and conversation history."
            else:
                mock_content = "This is a mock LLM response. In a real implementation, this would be generated by an actual language model."
            
            # Mock usage statistics
            usage = LLMUsage(
                prompt_tokens=sum(len(msg.content.split()) for msg in messages),
                completion_tokens=len(mock_content.split()),
                total_tokens=sum(len(msg.content.split()) for msg in messages) + len(mock_content.split())
            )
            
            if stream:
                return self._mock_streaming_response(mock_content, span)
            else:
                # Record metrics
                LLM_TOKENS.labels(
                    provider="mock",
                    type="prompt"
                ).inc(usage.prompt_tokens)
                
                LLM_TOKENS.labels(
                    provider="mock",
                    type="completion"
                ).inc(usage.completion_tokens)
                
                response = LLMResponse(
                    content=mock_content,
                    usage=usage,
                    model=self.model,
                    finish_reason="stop",
                    metadata={
                        "estimated_cost_cents": 0.0,
                        "provider": "mock"
                    }
                )
                
                span.set_attribute("success", True)
                span.set_attribute("total_tokens", usage.total_tokens)
                
                return response
    
    async def _mock_streaming_response(
        self,
        content: str,
        span
    ) -> AsyncGenerator[str, None]:
        """Generate mock streaming response.
        
        Args:
            content: Content to stream
            span: OpenTelemetry span
            
        Yields:
            Content chunks
        """
        import asyncio
        
        words = content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay
        
        span.set_attribute("success", True)
        span.set_attribute("streaming", True)
        span.set_attribute("word_count", len(words))
    
    async def close(self) -> None:
        """Close the mock provider (no-op)."""
        logger.info("Mock LLM provider closed")
