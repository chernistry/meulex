"""Unified exception handling for Meulex."""

from typing import Any, Dict, Optional


class MeulexException(Exception):
    """Base exception for Meulex."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(MeulexException):
    """Input validation error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details
        )


class AuthenticationError(MeulexException):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(MeulexException):
    """Authorization error."""
    
    def __init__(self, message: str = "Access denied") -> None:
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class NotFoundError(MeulexException):
    """Resource not found error."""
    
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(
            message=message,
            error_code="NOT_FOUND_ERROR",
            status_code=404
        )


class RateLimitError(MeulexException):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429
        )


class ExternalServiceError(MeulexException):
    """External service error."""
    
    def __init__(
        self,
        message: str,
        service: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={**(details or {}), "service": service}
        )


class LLMError(ExternalServiceError):
    """LLM provider error."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            service=provider,
            details=details
        )
        self.error_code = "LLM_ERROR"


class EmbeddingError(ExternalServiceError):
    """Embedding provider error."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            service=provider,
            details=details
        )
        self.error_code = "EMBEDDING_ERROR"


class VectorStoreError(ExternalServiceError):
    """Vector store error."""
    
    def __init__(
        self,
        message: str,
        store: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            service=store,
            details=details
        )
        self.error_code = "VECTOR_STORE_ERROR"


class BudgetExceededError(MeulexException):
    """Budget exceeded error."""
    
    def __init__(
        self,
        message: str = "Budget exceeded",
        budget_type: str = "tokens",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="BUDGET_EXCEEDED_ERROR",
            status_code=429,
            details={**(details or {}), "budget_type": budget_type}
        )


class SlackError(MeulexException):
    """Slack integration error."""
    
    def __init__(
        self,
        message: str,
        service: str = "slack",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "SLACK_ERROR", service, details)


class CircuitBreakerOpenError(MeulexException):
    """Circuit breaker open error."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        component: str = "unknown"
    ) -> None:
        super().__init__(
            message=message,
            error_code="CIRCUIT_BREAKER_OPEN_ERROR",
            status_code=503,
            details={"component": component}
        )


def to_error_response(
    exception: Exception,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convert exception to error response format.
    
    Args:
        exception: Exception to convert
        request_id: Optional request ID
        
    Returns:
        Error response dictionary
    """
    if isinstance(exception, MeulexException):
        return {
            "error_code": exception.error_code,
            "message": exception.message,
            "status_code": exception.status_code,
            "details": exception.details,
            "request_id": request_id
        }
    
    # Handle unknown exceptions
    return {
        "error_code": "INTERNAL_ERROR",
        "message": "An internal error occurred",
        "status_code": 500,
        "details": {"original_error": str(exception)},
        "request_id": request_id
    }
