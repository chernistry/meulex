"""Enhanced security middleware for Meulex."""

import logging
import re
import time
from typing import Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from meulex.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add comprehensive security headers."""
    
    def __init__(self, app, settings: Settings = None):
        """Initialize security headers middleware.
        
        Args:
            app: FastAPI application
            settings: Application settings
        """
        super().__init__(app)
        self.settings = settings or get_settings()
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to all responses.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response with security headers
        """
        response = await call_next(request)
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Additional security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS for HTTPS (only add if using HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, settings: Settings = None):
        """Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            settings: Application settings
        """
        super().__init__(app)
        self.settings = settings or get_settings()
        self.enabled = settings.enable_rate_limiting
        
        # Rate limit configurations per endpoint
        self.rate_limits = {
            "/chat": {"requests": 10, "window": 60},  # 10 requests per minute
            "/embed": {"requests": 20, "window": 60},  # 20 requests per minute
            "/slack/events": {"requests": 100, "window": 60},  # 100 requests per minute
            "default": {"requests": 60, "window": 60}  # 60 requests per minute for others
        }
        
        # In-memory storage for rate limiting
        # In production, use Redis or similar distributed cache
        self.request_counts: Dict[str, Dict[str, int]] = {}
        self.window_starts: Dict[str, float] = {}
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting.
        
        Args:
            request: HTTP request
            
        Returns:
            Client identifier
        """
        # Use X-Forwarded-For if behind proxy, otherwise client IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return client_ip
    
    def _get_rate_limit_config(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for path.
        
        Args:
            path: Request path
            
        Returns:
            Rate limit configuration
        """
        # Check for exact match first
        if path in self.rate_limits:
            return self.rate_limits[path]
        
        # Check for pattern matches
        for pattern, config in self.rate_limits.items():
            if pattern != "default" and re.match(pattern.replace("*", ".*"), path):
                return config
        
        return self.rate_limits["default"]
    
    def _is_rate_limited(self, client_id: str, path: str) -> tuple[bool, Dict[str, int]]:
        """Check if client is rate limited.
        
        Args:
            client_id: Client identifier
            path: Request path
            
        Returns:
            Tuple of (is_limited, rate_limit_info)
        """
        config = self._get_rate_limit_config(path)
        key = f"{client_id}:{path}"
        current_time = time.time()
        
        # Initialize if not exists
        if key not in self.window_starts:
            self.window_starts[key] = current_time
            self.request_counts[key] = {"count": 0}
        
        # Check if window has expired
        if current_time - self.window_starts[key] >= config["window"]:
            # Reset window
            self.window_starts[key] = current_time
            self.request_counts[key] = {"count": 0}
        
        # Increment request count
        self.request_counts[key]["count"] += 1
        current_count = self.request_counts[key]["count"]
        
        # Check if rate limited
        is_limited = current_count > config["requests"]
        
        rate_limit_info = {
            "limit": config["requests"],
            "remaining": max(0, config["requests"] - current_count),
            "reset": int(self.window_starts[key] + config["window"]),
            "window": config["window"]
        }
        
        return is_limited, rate_limit_info
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response or rate limit error
        """
        if not self.enabled:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        path = request.url.path
        
        # Skip rate limiting for health checks and metrics
        if path.startswith(("/health", "/metrics", "/info")):
            return await call_next(request)
        
        is_limited, rate_limit_info = self._is_rate_limited(client_id, path)
        
        if is_limited:
            logger.warning(
                f"Rate limit exceeded for client {client_id} on path {path}",
                extra={
                    "client_id": client_id,
                    "path": path,
                    "limit": rate_limit_info["limit"],
                    "window": rate_limit_info["window"]
                }
            )
            
            # Return rate limit error
            response = Response(
                content='{"error": "Rate limit exceeded", "message": "Too many requests"}',
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_limit_info["reset"]),
                    "Retry-After": str(rate_limit_info["window"])
                }
            )
            return response
        
        # Process request normally
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_info["reset"])
        
        return response


class LogSanitizerMiddleware(BaseHTTPMiddleware):
    """Middleware to sanitize sensitive information from logs."""
    
    def __init__(self, app, settings: Settings = None):
        """Initialize log sanitizer middleware.
        
        Args:
            app: FastAPI application
            settings: Application settings
        """
        super().__init__(app)
        self.settings = settings or get_settings()
        
        # Patterns to sanitize from logs
        self.sensitive_patterns = [
            # API keys and tokens
            (re.compile(r'(api[_-]?key|token|secret|password)["\s]*[:=]["\s]*([^"\s,}]+)', re.IGNORECASE), r'\1": "[REDACTED]"'),
            # Authorization headers
            (re.compile(r'(authorization|x-api-key)["\s]*:["\s]*([^"\s,}]+)', re.IGNORECASE), r'\1": "[REDACTED]"'),
            # Slack signatures
            (re.compile(r'(x-slack-signature)["\s]*:["\s]*([^"\s,}]+)', re.IGNORECASE), r'\1": "[REDACTED]"'),
            # Email addresses
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL_REDACTED]'),
            # Phone numbers (simple pattern)
            (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE_REDACTED]'),
            # Credit card numbers (simple pattern)
            (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '[CARD_REDACTED]'),
        ]
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize sensitive information from text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        for pattern, replacement in self.sensitive_patterns:
            text = pattern.sub(replacement, text)
        return text
    
    async def dispatch(self, request: Request, call_next):
        """Sanitize request/response data in logs.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Store original logging methods
        original_info = logger.info
        original_warning = logger.warning
        original_error = logger.error
        
        def sanitized_log(level_func):
            """Create sanitized logging function."""
            def log_func(msg, *args, **kwargs):
                if isinstance(msg, str):
                    msg = self._sanitize_text(msg)
                
                # Sanitize extra data
                if 'extra' in kwargs and isinstance(kwargs['extra'], dict):
                    sanitized_extra = {}
                    for key, value in kwargs['extra'].items():
                        if isinstance(value, str):
                            sanitized_extra[key] = self._sanitize_text(value)
                        else:
                            sanitized_extra[key] = value
                    kwargs['extra'] = sanitized_extra
                
                return level_func(msg, *args, **kwargs)
            return log_func
        
        # Temporarily replace logging methods
        logger.info = sanitized_log(original_info)
        logger.warning = sanitized_log(original_warning)
        logger.error = sanitized_log(original_error)
        
        try:
            response = await call_next(request)
            return response
        finally:
            # Restore original logging methods
            logger.info = original_info
            logger.warning = original_warning
            logger.error = original_error
