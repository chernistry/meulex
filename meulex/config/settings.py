"""Configuration settings for Meulex using Pydantic v2."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main configuration settings for Meulex."""
    
    # =============================================================================
    # Core Service Configuration
    # =============================================================================
    service_name: str = Field(default="meulex", description="Service name")
    service_version: str = Field(default="0.1.0", description="Service version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # =============================================================================
    # LLM Provider Configuration
    # =============================================================================
    llm_provider: str = Field(default="openai", description="Primary LLM provider")
    llm_model: str = Field(default="gpt-3.5-turbo", description="Primary LLM model")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI base URL"
    )
    openai_organization: Optional[str] = Field(
        default=None, description="OpenAI organization"
    )
    
    # Fallback LLM Provider
    fallback_llm_provider: str = Field(
        default="mock", description="Fallback LLM provider"
    )
    fallback_llm_model: str = Field(
        default="llama2", description="Fallback LLM model"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    
    # LLM Configuration
    max_tokens: int = Field(default=2048, description="Maximum tokens per request")
    request_timeout_s: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    cost_budget_cents: float = Field(
        default=1000.0, description="Cost budget in cents per request"
    )
    temperature: float = Field(default=0.7, description="LLM temperature")
    
    # =============================================================================
    # Embeddings Configuration
    # =============================================================================
    embedder_name: str = Field(default="jina", description="Embedder provider name")
    jina_api_key: Optional[str] = Field(default=None, description="Jina API key")
    jina_model: str = Field(
        default="jina-embeddings-v2-base-en", description="Jina model name"
    )
    embedding_dimension: int = Field(
        default=768, description="Embedding vector dimension"
    )
    embedding_batch_size: int = Field(
        default=64, description="Embedding batch size"
    )
    
    # =============================================================================
    # Vector Store Configuration
    # =============================================================================
    vector_store: str = Field(default="qdrant", description="Vector store provider")
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant URL"
    )
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(
        default="meulex_docs", description="Vector collection name"
    )
    vector_size: int = Field(default=768, description="Vector size")
    
    # =============================================================================
    # Caching Configuration
    # =============================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis URL"
    )
    cache_ttl_seconds: int = Field(
        default=3600, description="Default cache TTL in seconds"
    )
    semantic_cache_ttl_seconds: int = Field(
        default=1800, description="Semantic cache TTL in seconds"
    )
    enable_cache: bool = Field(default=True, description="Enable caching")
    
    # =============================================================================
    # Retrieval Configuration
    # =============================================================================
    enable_hybrid_retrieval: bool = Field(
        default=True, description="Enable hybrid retrieval"
    )
    enable_sparse_retrieval: bool = Field(
        default=True, description="Enable sparse retrieval"
    )
    rrf_k: int = Field(default=60, description="RRF parameter k")
    dense_weight: float = Field(default=0.7, description="Dense retrieval weight")
    sparse_weight: float = Field(default=0.3, description="Sparse retrieval weight")
    
    # Retrieval Parameters
    default_top_k: int = Field(default=3, description="Default top-k for retrieval")
    max_top_k: int = Field(default=20, description="Maximum top-k for retrieval")
    chunk_size: int = Field(default=512, description="Text chunk size")
    chunk_overlap: int = Field(default=64, description="Text chunk overlap")
    
    # Reranking
    enable_reranker: bool = Field(default=False, description="Enable reranking")
    reranker_model: str = Field(
        default="jina-reranker-v1-base-en", description="Reranker model"
    )
    reranker_top_k: int = Field(default=5, description="Reranker top-k")
    
    # =============================================================================
    # Security Configuration
    # =============================================================================
    rate_limits_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60, description="Rate limit requests per minute"
    )
    rate_limit_burst: int = Field(default=10, description="Rate limit burst")
    
    enable_security_headers: bool = Field(
        default=True, description="Enable security headers"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True, description="CORS allow credentials"
    )
    
    metrics_protected: bool = Field(
        default=False, description="Protect metrics endpoint"
    )
    metrics_token: Optional[str] = Field(
        default=None, description="Metrics access token"
    )
    
    # =============================================================================
    # Observability Configuration
    # =============================================================================
    metrics_enabled: bool = Field(default=True, description="Enable metrics")
    prometheus_port: int = Field(default=8001, description="Prometheus port")
    
    tracing_enabled: bool = Field(default=True, description="Enable tracing")
    otel_service_name: str = Field(default="meulex", description="OTel service name")
    otel_service_version: str = Field(
        default="0.1.0", description="OTel service version"
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTel OTLP endpoint"
    )
    jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces", description="Jaeger endpoint"
    )
    
    # Langfuse
    langfuse_enabled: bool = Field(default=False, description="Enable Langfuse")
    langfuse_secret_key: Optional[str] = Field(
        default=None, description="Langfuse secret key"
    )
    langfuse_public_key: Optional[str] = Field(
        default=None, description="Langfuse public key"
    )
    langfuse_host: str = Field(
        default="http://localhost:3000", description="Langfuse host"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    enable_log_sanitization: bool = Field(
        default=True, description="Enable log sanitization"
    )
    
    # =============================================================================
    # Security Settings
    # =============================================================================
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    enable_log_sanitization: bool = Field(default=True, description="Enable log sanitization")
    metrics_token: Optional[str] = Field(None, description="Token to protect metrics endpoint")
    
    # =============================================================================
    # Slack Integration
    # =============================================================================
    slack_bot_token: Optional[str] = Field(None, description="Slack bot token")
    slack_signing_secret: Optional[str] = Field(None, description="Slack signing secret")
    slack_bot_user_id: Optional[str] = Field(None, description="Slack bot user ID")
    slack_signature_verification_enabled: bool = Field(default=True, description="Enable Slack signature verification")
    
    # =============================================================================
    # Feature Flags
    # =============================================================================
    enable_reranker: bool = Field(default=False, description="Enable reranking")
    reranker_name: str = Field(default="keyword", description="Reranker to use")
    use_mock_reranker: bool = Field(default=False, description="Use mock reranker")
    enable_streaming: bool = Field(default=False, description="Enable streaming")
    enable_pii_masking: bool = Field(default=True, description="Enable PII masking")
    enable_circuit_breaker: bool = Field(
        default=True, description="Enable circuit breaker"
    )
    enable_budget_enforcement: bool = Field(
        default=True, description="Enable budget enforcement"
    )
    
    # =============================================================================
    # Ingestion Configuration
    # =============================================================================
    prefect_api_url: str = Field(
        default="http://localhost:4200/api", description="Prefect API URL"
    )
    prefect_logging_level: str = Field(
        default="INFO", description="Prefect logging level"
    )
    
    supported_file_types: List[str] = Field(
        default=["txt", "md", "pdf", "docx", "html"],
        description="Supported file types"
    )
    max_file_size_mb: int = Field(
        default=10, description="Maximum file size in MB"
    )
    max_content_length: int = Field(
        default=50000, description="Maximum content length"
    )
    
    # =============================================================================
    # Development Configuration
    # =============================================================================
    use_mock_llm: bool = Field(default=False, description="Use mock LLM")
    use_mock_embeddings: bool = Field(
        default=False, description="Use mock embeddings"
    )
    use_mock_reranker: bool = Field(default=False, description="Use mock reranker")
    
    # Testing
    test_collection_name: str = Field(
        default="meulex_test_docs", description="Test collection name"
    )
    test_redis_db: int = Field(default=1, description="Test Redis database")
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator("default_top_k", "max_top_k", "reranker_top_k")
    def validate_top_k(cls, v: int) -> int:
        """Validate top-k values."""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v
    
    @validator("dense_weight", "sparse_weight")
    def validate_weights(cls, v: float) -> float:
        """Validate retrieval weights."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
