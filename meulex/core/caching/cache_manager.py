"""Simple cache manager for Meulex."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

import redis

from meulex.config.settings import Settings
from meulex.observability import CACHE_HITS, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class CacheManager:
    """Simple cache manager with Redis backend."""
    
    def __init__(self, settings: Settings):
        """Initialize cache manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.enabled = settings.enable_cache
        self.default_ttl = settings.cache_ttl_seconds
        self.semantic_ttl = settings.semantic_cache_ttl_seconds
        
        # Redis client
        self.redis_client = None
        if self.enabled:
            try:
                self.redis_client = redis.from_url(settings.redis_url)
                self.redis_client.ping()
                logger.info("Cache manager initialized with Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self.enabled = False
        
        if not self.enabled:
            logger.info("Cache manager initialized (disabled)")
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate cache key from data.
        
        Args:
            prefix: Key prefix
            data: Data to hash
            
        Returns:
            Cache key
        """
        # Create deterministic hash from data
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{prefix}:{data_hash}"
    
    def generate_semantic_cache_key(
        self,
        question: str,
        top_k: int,
        mode: str = "balanced",
        provider: str = "default"
    ) -> str:
        """Generate semantic cache key for chat queries.
        
        Args:
            question: User question
            top_k: Number of documents requested
            mode: Generation mode
            provider: LLM provider
            
        Returns:
            Semantic cache key
        """
        # Normalize question for better cache hits
        normalized_question = question.strip().lower()
        
        cache_data = {
            "question": normalized_question,
            "top_k": top_k,
            "mode": mode,
            "provider": provider,
            "version": "v1"  # For cache invalidation
        }
        
        return self._generate_cache_key("semantic", cache_data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        with tracer.start_as_current_span("cache_get") as span:
            span.set_attribute("cache_key", key)
            
            try:
                value = self.redis_client.get(key)
                
                if value is not None:
                    # Record cache hit
                    CACHE_HITS.labels(type="redis", result="hit").inc()
                    span.set_attribute("cache_hit", True)
                    
                    # Deserialize JSON
                    return json.loads(value.decode('utf-8'))
                else:
                    # Record cache miss
                    CACHE_HITS.labels(type="redis", result="miss").inc()
                    span.set_attribute("cache_hit", False)
                    return None
                    
            except Exception as e:
                logger.warning(f"Cache get failed for key {key}: {e}")
                span.set_attribute("error", str(e))
                return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.redis_client:
            return False
        
        with tracer.start_as_current_span("cache_set") as span:
            span.set_attribute("cache_key", key)
            span.set_attribute("ttl", ttl or self.default_ttl)
            
            try:
                # Serialize to JSON
                serialized_value = json.dumps(value, separators=(',', ':'))
                
                # Set with TTL
                ttl = ttl or self.default_ttl
                success = self.redis_client.setex(
                    key,
                    ttl,
                    serialized_value
                )
                
                span.set_attribute("success", bool(success))
                return bool(success)
                
            except Exception as e:
                logger.warning(f"Cache set failed for key {key}: {e}")
                span.set_attribute("error", str(e))
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            deleted = self.redis_client.delete(key)
            return bool(deleted)
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern.
        
        Args:
            pattern: Key pattern (with wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Cache clear pattern failed for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        stats = {
            "enabled": self.enabled,
            "default_ttl": self.default_ttl,
            "semantic_ttl": self.semantic_ttl
        }
        
        if self.enabled and self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_connected": True,
                    "redis_memory_used": info.get("used_memory_human", "unknown"),
                    "redis_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0
                })
            except Exception as e:
                stats["redis_connected"] = False
                stats["redis_error"] = str(e)
        else:
            stats["redis_connected"] = False
        
        return stats
    
    async def close(self) -> None:
        """Close cache connections."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Cache manager closed")
            except Exception as e:
                logger.warning(f"Error closing cache manager: {e}")
