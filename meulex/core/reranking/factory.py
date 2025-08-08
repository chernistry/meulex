"""Reranker factory for creating reranker instances."""

import logging
from typing import Dict, Optional, Type

from meulex.config.settings import Settings
from meulex.core.reranking.base import BaseReranker
from meulex.core.reranking.simple import KeywordReranker, MockReranker
from meulex.utils.exceptions import MeulexException

logger = logging.getLogger(__name__)

# Registry of available rerankers
RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "keyword": KeywordReranker,
    "mock": MockReranker,
}


def create_reranker(settings: Settings, reranker_name: Optional[str] = None) -> Optional[BaseReranker]:
    """Create a reranker instance based on settings.
    
    Args:
        settings: Application settings
        reranker_name: Optional reranker name override
        
    Returns:
        Reranker instance or None if disabled
        
    Raises:
        MeulexException: If reranker type is not supported
    """
    # Check if reranking is enabled
    if not settings.enable_reranker:
        logger.info("Reranking disabled in settings")
        return None
    
    reranker_name = reranker_name or settings.reranker_name.lower()
    
    # Use mock reranker if explicitly requested or in development mode
    if settings.use_mock_reranker or reranker_name == "mock":
        reranker_name = "mock"
    
    if reranker_name not in RERANKER_REGISTRY:
        available = ", ".join(RERANKER_REGISTRY.keys())
        raise MeulexException(
            f"Unsupported reranker: {reranker_name}. Available: {available}",
            "RERANKER_ERROR",
            500,
            {"available_rerankers": list(RERANKER_REGISTRY.keys())}
        )
    
    reranker_class = RERANKER_REGISTRY[reranker_name]
    
    try:
        reranker = reranker_class(settings)
        logger.info(
            f"Created reranker: {reranker_name}",
            extra={
                "reranker": reranker_name,
                "reranker_name": reranker.name
            }
        )
        return reranker
        
    except Exception as e:
        logger.error(f"Failed to create reranker {reranker_name}: {e}")
        raise MeulexException(
            f"Failed to create reranker {reranker_name}",
            "RERANKER_ERROR",
            500,
            {"error": str(e), "reranker": reranker_name}
        )


def list_available_rerankers() -> Dict[str, str]:
    """List available rerankers.
    
    Returns:
        Dictionary mapping reranker names to descriptions
    """
    return {
        "keyword": "Simple keyword-based reranking",
        "mock": "Mock reranker for development/testing"
    }
