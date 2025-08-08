"""Embeddings factory for creating embedder instances."""

import logging
from typing import Dict, Type

from meulex.config.settings import Settings
from meulex.core.embeddings.base import BaseEmbedder
from meulex.core.embeddings.providers.jina import JinaEmbedder, MockEmbedder
from meulex.utils.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Registry of available embedders
EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "jina": JinaEmbedder,
    "mock": MockEmbedder,
}


def create_embedder(settings: Settings) -> BaseEmbedder:
    """Create an embedder instance based on settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Embedder instance
        
    Raises:
        EmbeddingError: If embedder type is not supported
    """
    embedder_name = settings.embedder_name.lower()
    
    # Use mock embedder if explicitly requested or if in development mode
    if settings.use_mock_embeddings or embedder_name == "mock":
        embedder_name = "mock"
    
    if embedder_name not in EMBEDDER_REGISTRY:
        available = ", ".join(EMBEDDER_REGISTRY.keys())
        raise EmbeddingError(
            f"Unsupported embedder: {embedder_name}. Available: {available}",
            "factory"
        )
    
    embedder_class = EMBEDDER_REGISTRY[embedder_name]
    
    try:
        embedder = embedder_class(settings)
        logger.info(
            f"Created embedder: {embedder_name}",
            extra={
                "embedder": embedder_name,
                "model": embedder.model_name,
                "dimension": embedder.dimension
            }
        )
        return embedder
        
    except Exception as e:
        logger.error(f"Failed to create embedder {embedder_name}: {e}")
        raise EmbeddingError(
            f"Failed to create embedder {embedder_name}",
            "factory",
            {"error": str(e), "embedder": embedder_name}
        )


def list_available_embedders() -> Dict[str, str]:
    """List available embedders.
    
    Returns:
        Dictionary mapping embedder names to descriptions
    """
    return {
        "jina": "Jina AI embeddings API",
        "mock": "Mock embedder for development/testing"
    }
