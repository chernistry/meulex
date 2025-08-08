"""Slack authentication and signature verification."""

import hashlib
import hmac
import logging
import time
from typing import Optional

from meulex.config.settings import Settings
from meulex.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class SlackSignatureVerifier:
    """Slack request signature verifier."""
    
    def __init__(self, settings: Settings):
        """Initialize signature verifier.
        
        Args:
            settings: Application settings
        """
        self.signing_secret = settings.slack_signing_secret
        self.request_timeout = 300  # 5 minutes
        
        if not self.signing_secret:
            logger.warning("Slack signing secret not configured - signature verification disabled")
    
    def verify_signature(
        self,
        timestamp: str,
        body: bytes,
        signature: str
    ) -> bool:
        """Verify Slack request signature.
        
        Args:
            timestamp: X-Slack-Request-Timestamp header
            body: Raw request body
            signature: X-Slack-Signature header
            
        Returns:
            True if signature is valid
            
        Raises:
            AuthenticationError: If signature verification fails
        """
        if not self.signing_secret:
            logger.warning("Signature verification skipped - no signing secret configured")
            return True
        
        try:
            # Check timestamp to prevent replay attacks
            request_timestamp = int(timestamp)
            current_timestamp = int(time.time())
            
            if abs(current_timestamp - request_timestamp) > self.request_timeout:
                raise AuthenticationError(
                    f"Request timestamp too old: {current_timestamp - request_timestamp}s > {self.request_timeout}s"
                )
            
            # Create signature base string
            sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
            
            # Calculate expected signature
            expected_signature = 'v0=' + hmac.new(
                self.signing_secret.encode('utf-8'),
                sig_basestring.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures using constant-time comparison
            if not hmac.compare_digest(expected_signature, signature):
                raise AuthenticationError("Invalid signature")
            
            logger.debug("Slack signature verified successfully")
            return True
            
        except ValueError as e:
            raise AuthenticationError(f"Invalid timestamp format: {e}")
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            raise AuthenticationError("Signature verification failed")
    
    def extract_headers(self, headers: dict) -> tuple[Optional[str], Optional[str]]:
        """Extract Slack headers from request.
        
        Args:
            headers: Request headers
            
        Returns:
            Tuple of (timestamp, signature)
        """
        # Headers can be in different cases
        timestamp = None
        signature = None
        
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower == 'x-slack-request-timestamp':
                timestamp = value
            elif key_lower == 'x-slack-signature':
                signature = value
        
        return timestamp, signature
