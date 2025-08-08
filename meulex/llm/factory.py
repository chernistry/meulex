"""LLM factory for creating provider instances."""

import logging
from typing import Dict, Optional, Type

from meulex.config.settings import Settings
from meulex.llm.base import BaseLLMProvider
from meulex.llm.providers.openai import MockLLMProvider, OpenAIProvider
from meulex.utils.exceptions import LLMError

logger = logging.getLogger(__name__)

# Registry of available LLM providers
LLM_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "mock": MockLLMProvider,
}


def create_llm_provider(settings: Settings, provider_name: Optional[str] = None) -> BaseLLMProvider:
    """Create an LLM provider instance based on settings.
    
    Args:
        settings: Application settings
        provider_name: Optional provider name override
        
    Returns:
        LLM provider instance
        
    Raises:
        LLMError: If provider type is not supported
    """
    provider_name = provider_name or settings.llm_provider.lower()
    
    # Use mock provider if explicitly requested or if in development mode
    if settings.use_mock_llm or provider_name == "mock":
        provider_name = "mock"
    
    if provider_name not in LLM_PROVIDER_REGISTRY:
        available = ", ".join(LLM_PROVIDER_REGISTRY.keys())
        raise LLMError(
            f"Unsupported LLM provider: {provider_name}. Available: {available}",
            "factory"
        )
    
    provider_class = LLM_PROVIDER_REGISTRY[provider_name]
    
    try:
        provider = provider_class(settings)
        # logger.info(
        #     f"Created LLM provider: {provider_name}",
        #     extra={
        #         "provider": provider_name,
        #         "model": provider.default_model,
        #         "provider_name": provider.name
        #     }
        # )
        logger.info(f"Created LLM provider: {provider_name} with model: {provider.default_model}")
        return provider
        
    except Exception as e:
        logger.error(f"Failed to create LLM provider {provider_name}: {e}")
        raise LLMError(
            f"Failed to create LLM provider {provider_name}",
            "factory",
            {"error": str(e), "provider": provider_name}
        )


def list_available_providers() -> Dict[str, str]:
    """List available LLM providers.
    
    Returns:
        Dictionary mapping provider names to descriptions
    """
    return {
        "openai": "OpenAI-compatible API (GPT models)",
        "mock": "Mock provider for development/testing"
    }
