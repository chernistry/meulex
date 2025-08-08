"""Jina embeddings provider."""

import asyncio
import logging
from typing import List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from meulex.config.settings import Settings
from meulex.core.embeddings.base import BaseEmbedder
from meulex.observability import EMBEDDINGS_GENERATED, get_tracer
from meulex.utils.exceptions import EmbeddingError

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class JinaEmbedder(BaseEmbedder):
    """Jina embeddings provider."""
    
    def __init__(self, settings: Settings) -> None:
        """Initialize Jina embedder.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.api_key = settings.jina_api_key
        self.model = settings.jina_model
        self.batch_size = settings.embedding_batch_size
        self.dimension_size = settings.embedding_dimension
        
        if not self.api_key:
            raise EmbeddingError(
                "Jina API key is required",
                "jina",
                {"model": self.model}
            )
        
        # HTTP client
        self.client = httpx.AsyncClient(
            base_url="https://api.jina.ai/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=settings.request_timeout_s
        )
        
        logger.info(
            "Jina embedder initialized",
            extra={
                "model": self.model,
                "batch_size": self.batch_size,
                "dimension": self.dimension_size
            }
        )
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension_size
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model
    
    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        with tracer.start_as_current_span("jina_embed_batch") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("batch_size", len(texts))
            
            try:
                payload = {
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float"
                }
                
                response = await self.client.post("/embeddings", json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract embeddings
                embeddings = []
                for item in data["data"]:
                    embeddings.append(item["embedding"])
                
                # Record metrics
                EMBEDDINGS_GENERATED.labels(
                    model=self.model,
                    provider="jina"
                ).inc(len(texts))
                
                logger.debug(
                    f"Generated {len(embeddings)} embeddings",
                    extra={
                        "model": self.model,
                        "batch_size": len(texts),
                        "dimension": len(embeddings[0]) if embeddings else 0
                    }
                )
                
                span.set_attribute("success", True)
                span.set_attribute("embeddings_count", len(embeddings))
                
                return embeddings
                
            except httpx.HTTPStatusError as e:
                error_msg = f"Jina API error: {e.response.status_code}"
                try:
                    error_detail = e.response.json()
                    error_msg += f" - {error_detail}"
                except Exception:
                    error_msg += f" - {e.response.text}"
                
                logger.error(error_msg)
                span.set_attribute("error", error_msg)
                
                raise EmbeddingError(
                    error_msg,
                    "jina",
                    {
                        "status_code": e.response.status_code,
                        "batch_size": len(texts),
                        "model": self.model
                    }
                )
                
            except Exception as e:
                error_msg = f"Failed to generate embeddings: {e}"
                logger.error(error_msg)
                span.set_attribute("error", error_msg)
                
                raise EmbeddingError(
                    error_msg,
                    "jina",
                    {
                        "batch_size": len(texts),
                        "model": self.model,
                        "error_type": type(e).__name__
                    }
                )
    
    async def embed_async_single(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            raise EmbeddingError(
                "Cannot embed empty text",
                "jina"
            )
        
        embeddings = await self._embed_batch([text])
        return embeddings[0]
    
    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            raise EmbeddingError(
                "Cannot embed empty texts",
                "jina"
            )
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(non_empty_texts), self.batch_size):
            batch = non_empty_texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid rate limiting
            if i + self.batch_size < len(non_empty_texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    async def close(self) -> None:
        """Close the HTTP client."""
        try:
            await self.client.aclose()
            logger.info("Jina embedder client closed")
        except Exception as e:
            logger.warning(f"Error closing Jina client: {e}")


class MockEmbedder(BaseEmbedder):
    """Mock embedder for development/testing."""
    
    def __init__(self, settings: Settings) -> None:
        """Initialize mock embedder.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.dimension_size = settings.embedding_dimension
        
        logger.info(
            "Mock embedder initialized",
            extra={"dimension": self.dimension_size}
        )
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension_size
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return "mock-embedder"
    
    async def embed_async_single(self, text: str) -> List[float]:
        """Embed a single text with mock data.
        
        Args:
            text: Text to embed
            
        Returns:
            Mock embedding vector
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text", "mock")
        
        # Generate deterministic mock embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and normalize
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.dimension_size:
            embedding.extend(embedding[:self.dimension_size - len(embedding)])
        
        embedding = embedding[:self.dimension_size]
        
        # Record metrics
        EMBEDDINGS_GENERATED.labels(
            model="mock",
            provider="mock"
        ).inc(1)
        
        return embedding
    
    async def embed_async_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with mock data.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of mock embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed_async_single(text)
            embeddings.append(embedding)
        
        return embeddings
    
    async def close(self) -> None:
        """Close the mock embedder (no-op)."""
        logger.info("Mock embedder closed")
