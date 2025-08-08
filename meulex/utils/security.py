"""Security utilities for Meulex."""

import hashlib
import hmac
import re
import time
from typing import Any, Dict, List, Optional

from meulex.utils.exceptions import AuthenticationError


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize log data by masking sensitive information.
    
    Args:
        data: Dictionary to sanitize
        
    Returns:
        Sanitized dictionary
    """
    sensitive_keys = {
        "password", "token", "key", "secret", "authorization", "cookie",
        "x-api-key", "x-auth-token", "bearer", "api_key", "access_token"
    }
    
    def _sanitize_value(key: str, value: Any) -> Any:
        if isinstance(key, str) and key.lower() in sensitive_keys:
            return "***MASKED***"
        
        if isinstance(value, str):
            # Mask potential tokens/keys in string values
            if re.match(r'^[A-Za-z0-9+/]{20,}={0,2}$', value):  # Base64-like
                return "***MASKED***"
            if re.match(r'^[a-f0-9]{32,}$', value):  # Hex strings
                return "***MASKED***"
            if re.match(r'^sk-[A-Za-z0-9]{20,}$', value):  # OpenAI-style keys
                return "***MASKED***"
        
        if isinstance(value, dict):
            return {k: _sanitize_value(k, v) for k, v in value.items()}
        
        if isinstance(value, list):
            return [_sanitize_value("", item) for item in value]
        
        return value
    
    return {k: _sanitize_value(k, v) for k, v in data.items()}


def mask_pii(text: str) -> str:
    """Mask PII in text.
    
    Args:
        text: Text to mask
        
    Returns:
        Text with PII masked
    """
    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '***EMAIL***',
        text
    )
    
    # Phone numbers (basic patterns)
    text = re.sub(
        r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        '***PHONE***',
        text
    )
    
    # Credit card numbers (basic pattern)
    text = re.sub(
        r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b',
        '***CARD***',
        text
    )
    
    # SSN (basic pattern)
    text = re.sub(
        r'\b[0-9]{3}-?[0-9]{2}-?[0-9]{4}\b',
        '***SSN***',
        text
    )
    
    return text


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
    max_age_seconds: int = 300
) -> bool:
    """Verify Slack request signature.
    
    Args:
        body: Request body bytes
        timestamp: Request timestamp
        signature: Slack signature header
        signing_secret: Slack signing secret
        max_age_seconds: Maximum age of request in seconds
        
    Returns:
        True if signature is valid
        
    Raises:
        AuthenticationError: If signature verification fails
    """
    try:
        # Check timestamp to prevent replay attacks
        current_time = int(time.time())
        request_time = int(timestamp)
        
        if abs(current_time - request_time) > max_age_seconds:
            raise AuthenticationError("Request timestamp too old")
        
        # Create signature base string
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        
        # Compute expected signature
        expected_signature = 'v0=' + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        if not hmac.compare_digest(expected_signature, signature):
            raise AuthenticationError("Invalid signature")
        
        return True
        
    except (ValueError, TypeError) as e:
        raise AuthenticationError(f"Signature verification failed: {e}")


def generate_request_id() -> str:
    """Generate a unique request ID.
    
    Returns:
        Unique request ID
    """
    import uuid
    return str(uuid.uuid4())


def validate_input_length(
    value: str,
    min_length: int = 1,
    max_length: int = 10000,
    field_name: str = "input"
) -> None:
    """Validate input string length.
    
    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    from meulex.utils.exceptions import ValidationError
    
    if len(value) < min_length:
        raise ValidationError(
            f"{field_name} must be at least {min_length} characters long"
        )
    
    if len(value) > max_length:
        raise ValidationError(
            f"{field_name} must be at most {max_length} characters long"
        )


def sanitize_html(text: str) -> str:
    """Basic HTML sanitization.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove JavaScript
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    # Remove common XSS patterns
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def get_security_headers() -> Dict[str, str]:
    """Get security headers for HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "media-src 'self'; "
            "frame-src 'none';"
        ),
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }
