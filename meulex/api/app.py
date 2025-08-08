"""FastAPI application for Meulex."""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import redis
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from meulex.config.settings import get_settings
from meulex.observability import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    instrument_fastapi,
    setup_observability,
)
from meulex.security.middleware import (
    LogSanitizerMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from meulex.utils.exceptions import MeulexException, to_error_response
from meulex.utils.security import generate_request_id, get_security_headers

logger = logging.getLogger(__name__)
settings = get_settings()

# Global Redis client for rate limiting
redis_client: redis.Redis = None


class RequestMiddleware(BaseHTTPMiddleware):
    """Middleware for request processing."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with timing and logging."""
        start_time = time.time()
        request_id = generate_request_id()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                endpoint=request.url.path,
                method=request.method,
                status=response.status_code,
                provider="unknown"
            ).inc()
            
            REQUEST_DURATION.labels(
                endpoint=request.url.path,
                method=request.method
            ).observe(duration)
            
            # Add security headers
            if settings.enable_security_headers:
                for key, value in get_security_headers().items():
                    response.headers[key] = value
            
            # Add request ID to response
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration": duration,
                }
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            REQUEST_COUNT.labels(
                endpoint=request.url.path,
                method=request.method,
                status=500,
                provider="unknown"
            ).inc()
            
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration": duration,
                },
                exc_info=True
            )
            
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, redis_client: redis.Redis = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.in_memory_cache = {}  # Fallback for when Redis is unavailable
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        if not settings.rate_limits_enabled:
            return await call_next(request)
        
        # Get client identifier (IP address for now)
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"
        
        try:
            if self.redis_client:
                # Use Redis for rate limiting
                current = self.redis_client.get(key)
                if current is None:
                    self.redis_client.setex(key, 60, 1)
                    current_count = 1
                else:
                    current_count = int(current)
                    if current_count >= settings.rate_limit_requests_per_minute:
                        raise HTTPException(
                            status_code=429,
                            detail="Rate limit exceeded"
                        )
                    self.redis_client.incr(key)
            else:
                # Fallback to in-memory rate limiting
                now = time.time()
                if key not in self.in_memory_cache:
                    self.in_memory_cache[key] = {"count": 1, "reset_time": now + 60}
                else:
                    cache_entry = self.in_memory_cache[key]
                    if now > cache_entry["reset_time"]:
                        cache_entry["count"] = 1
                        cache_entry["reset_time"] = now + 60
                    else:
                        if cache_entry["count"] >= settings.rate_limit_requests_per_minute:
                            raise HTTPException(
                                status_code=429,
                                detail="Rate limit exceeded"
                            )
                        cache_entry["count"] += 1
        
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Rate limiting failed: {e}")
            # Continue without rate limiting if there's an error
        
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global redis_client
    
    # Startup
    logger.info("Starting Meulex application")
    
    # Setup observability
    setup_observability(settings)
    
    # Setup Redis client
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meulex application")
    if redis_client:
        redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="Meulex",
    description="Slack-native, compliance-aware agentic RAG copilot",
    version=settings.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add security middleware
if settings.enable_log_sanitization:
    app.add_middleware(LogSanitizerMiddleware, settings=settings)

if settings.enable_rate_limiting:
    app.add_middleware(RateLimitMiddleware, settings=settings)

if settings.enable_security_headers:
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestMiddleware)
app.add_middleware(RateLimitMiddleware, redis_client=redis_client)

# Include API routes
from meulex.api.routes.embed import router as embed_router
from meulex.api.routes.chat import router as chat_router
from meulex.api.routes.slack import router as slack_router
app.include_router(embed_router)
app.include_router(chat_router)
app.include_router(slack_router)

# Instrument with OpenTelemetry
if settings.tracing_enabled:
    instrument_fastapi(app)


@app.exception_handler(MeulexException)
async def meulex_exception_handler(request: Request, exc: MeulexException):
    """Handle Meulex exceptions."""
    request_id = getattr(request.state, "request_id", None)
    error_response = to_error_response(exc, request_id)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code,
            "details": {},
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, "request_id", None)
    error_response = to_error_response(exc, request_id)
    
    logger.error(
        "Unhandled exception",
        extra={
            "request_id": request_id,
            "error": str(exc),
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )


@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "version": settings.service_version,
        "service": settings.service_name
    }


@app.get("/health/ready")
async def readiness_check():
    """Readiness check with dependency validation."""
    checks = {}
    overall_status = "ok"
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "unavailable"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        overall_status = "degraded"
    
    # Check Qdrant (basic connectivity)
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.qdrant_url}/health")
            if response.status_code == 200:
                checks["qdrant"] = "ok"
            else:
                checks["qdrant"] = f"error: status {response.status_code}"
                overall_status = "degraded"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "checks": checks
    }


@app.get("/health/live")
async def liveness_check():
    """Liveness check."""
    return {
        "status": "ok",
        "timestamp": time.time()
    }


@app.get("/info")
async def info():
    """Service information."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "environment": settings.environment,
        "features": {
            "hybrid_retrieval": settings.enable_hybrid_retrieval,
            "reranker": settings.enable_reranker,
            "streaming": settings.enable_streaming,
            "metrics": settings.metrics_enabled,
            "tracing": settings.tracing_enabled,
        },
        "providers": {
            "llm": settings.llm_provider,
            "embeddings": settings.embedder_name,
            "vector_store": settings.vector_store,
        }
    }


@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus metrics endpoint."""
    # Check if metrics are protected
    if settings.metrics_protected:
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = auth_header.split(" ")[1]
        if token != settings.metrics_token:
            raise HTTPException(status_code=403, detail="Invalid token")
    
    # Generate Prometheus metrics
    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "meulex.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
