"""LLM cascade with fallback and budget management."""

import logging
import time
from typing import Any, AsyncGenerator, List, Optional, Union

from meulex.config.settings import Settings
from meulex.llm.base import BaseLLMProvider, ChatMessage, LLMResponse
from meulex.llm.factory import create_llm_provider
from meulex.observability import LLM_FAILURES, get_tracer
from meulex.utils.exceptions import BudgetExceededError, CircuitBreakerOpenError, LLMError

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class CircuitBreaker:
    """Simple circuit breaker for LLM providers."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self) -> None:
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class LLMCascade:
    """LLM cascade with primary and fallback providers."""
    
    def __init__(self, settings: Settings):
        """Initialize LLM cascade.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Create primary provider
        self.primary_provider = create_llm_provider(settings)
        self.primary_circuit_breaker = CircuitBreaker()
        
        # Create fallback provider if configured
        self.fallback_provider = None
        self.fallback_circuit_breaker = None
        
        if settings.fallback_llm_provider and settings.fallback_llm_provider != settings.llm_provider:
            try:
                self.fallback_provider = create_llm_provider(
                    settings, 
                    settings.fallback_llm_provider
                )
                self.fallback_circuit_breaker = CircuitBreaker()
                logger.info(f"Fallback provider configured: {settings.fallback_llm_provider}")
            except Exception as e:
                logger.warning(f"Failed to create fallback provider: {e}")
        
        # Budget tracking
        self.total_cost_cents = 0.0
        self.request_count = 0
        
        logger.info(
            "LLM cascade initialized",
            extra={
                "primary_provider": self.primary_provider.name,
                "fallback_provider": self.fallback_provider.name if self.fallback_provider else None,
                "budget_enforcement": settings.enable_budget_enforcement
            }
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
        """Generate chat completion with cascade fallback.
        
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
        with tracer.start_as_current_span("llm_cascade_completion") as span:
            span.set_attribute("message_count", len(messages))
            span.set_attribute("temperature", temperature)
            span.set_attribute("stream", stream)
            
            # Check budget before proceeding
            if (self.settings.enable_budget_enforcement and 
                self.total_cost_cents >= self.settings.cost_budget_cents):
                raise BudgetExceededError(
                    f"Total budget exceeded: {self.total_cost_cents:.2f} cents",
                    "total",
                    {
                        "total_cost": self.total_cost_cents,
                        "budget": self.settings.cost_budget_cents,
                        "request_count": self.request_count
                    }
                )
            
            # Try primary provider first
            try:
                if self.primary_circuit_breaker.can_execute():
                    span.set_attribute("provider_used", "primary")
                    span.set_attribute("provider_name", self.primary_provider.name)
                    
                    response = await self.primary_provider.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        **kwargs
                    )
                    
                    # Record success and update budget
                    self.primary_circuit_breaker.record_success()
                    
                    if isinstance(response, LLMResponse) and response.usage:
                        estimated_cost = self.primary_provider.estimate_cost(
                            response.usage, 
                            response.model
                        )
                        self.total_cost_cents += estimated_cost
                        self.request_count += 1
                    
                    span.set_attribute("success", True)
                    return response
                    
                else:
                    raise CircuitBreakerOpenError(
                        "Primary provider circuit breaker is open",
                        self.primary_provider.name
                    )
                    
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
                self.primary_circuit_breaker.record_failure()
                
                # Record failure metrics
                LLM_FAILURES.labels(
                    provider=self.primary_provider.name,
                    error_type="primary_failure"
                ).inc()
                
                # Try fallback if available
                if self.fallback_provider and self.fallback_circuit_breaker:
                    try:
                        if self.fallback_circuit_breaker.can_execute():
                            logger.info("Attempting fallback provider")
                            span.set_attribute("provider_used", "fallback")
                            span.set_attribute("provider_name", self.fallback_provider.name)
                            span.set_attribute("primary_error", str(e))
                            
                            response = await self.fallback_provider.chat_completion(
                                messages=messages,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=stream,
                                **kwargs
                            )
                            
                            # Record success and update budget
                            self.fallback_circuit_breaker.record_success()
                            
                            if isinstance(response, LLMResponse) and response.usage:
                                estimated_cost = self.fallback_provider.estimate_cost(
                                    response.usage,
                                    response.model
                                )
                                self.total_cost_cents += estimated_cost
                                self.request_count += 1
                            
                            span.set_attribute("success", True)
                            span.set_attribute("fallback_used", True)
                            return response
                            
                        else:
                            raise CircuitBreakerOpenError(
                                "Fallback provider circuit breaker is open",
                                self.fallback_provider.name
                            )
                            
                    except Exception as fallback_error:
                        logger.error(f"Fallback provider also failed: {fallback_error}")
                        self.fallback_circuit_breaker.record_failure()
                        
                        # Record fallback failure metrics
                        LLM_FAILURES.labels(
                            provider=self.fallback_provider.name,
                            error_type="fallback_failure"
                        ).inc()
                        
                        # Re-raise the original error since both failed
                        span.set_attribute("error", str(e))
                        span.set_attribute("fallback_error", str(fallback_error))
                        raise e
                
                # No fallback available, re-raise original error
                span.set_attribute("error", str(e))
                raise e
    
    def get_status(self) -> dict:
        """Get cascade status information.
        
        Returns:
            Status dictionary
        """
        status = {
            "primary_provider": {
                "name": self.primary_provider.name,
                "model": self.primary_provider.default_model,
                "circuit_breaker_state": self.primary_circuit_breaker.state,
                "failure_count": self.primary_circuit_breaker.failure_count
            },
            "fallback_provider": None,
            "budget": {
                "total_cost_cents": self.total_cost_cents,
                "budget_cents": self.settings.cost_budget_cents,
                "request_count": self.request_count,
                "budget_remaining_cents": max(0, self.settings.cost_budget_cents - self.total_cost_cents)
            }
        }
        
        if self.fallback_provider:
            status["fallback_provider"] = {
                "name": self.fallback_provider.name,
                "model": self.fallback_provider.default_model,
                "circuit_breaker_state": self.fallback_circuit_breaker.state,
                "failure_count": self.fallback_circuit_breaker.failure_count
            }
        
        return status
    
    def reset_budget(self) -> None:
        """Reset budget tracking."""
        self.total_cost_cents = 0.0
        self.request_count = 0
        logger.info("Budget tracking reset")
    
    async def close(self) -> None:
        """Close all providers."""
        try:
            await self.primary_provider.close()
            if self.fallback_provider:
                await self.fallback_provider.close()
            logger.info("LLM cascade closed")
        except Exception as e:
            logger.warning(f"Error closing LLM cascade: {e}")
