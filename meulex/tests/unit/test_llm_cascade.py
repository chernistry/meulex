"""Unit tests for LLM cascade functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from meulex.config.settings import Settings
from meulex.llm.base import ChatMessage, LLMResponse, LLMUsage, MessageRole
from meulex.llm.cascade import CircuitBreaker, LLMCascade
from meulex.utils.exceptions import BudgetExceededError, CircuitBreakerOpenError, LLMError


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.can_execute() is True
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker tracks failures correctly."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        
        # First failure
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 1
        assert cb.can_execute() is True
        
        # Second failure - should open circuit
        cb.record_failure()
        assert cb.state == "open"
        assert cb.failure_count == 2
        assert cb.can_execute() is False
    
    def test_circuit_breaker_success_reset(self):
        """Test circuit breaker resets on success."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        
        # Add failure
        cb.record_failure()
        assert cb.failure_count == 1
        
        # Success should reset
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)  # 100ms timeout
        
        # Trigger circuit open
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_execute() is False
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should be half-open now
        assert cb.can_execute() is True
        
        # Success should close circuit
        cb.record_success()
        assert cb.state == "closed"


class TestLLMCascade:
    """Test LLM cascade functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = MagicMock(spec=Settings)
        settings.llm_provider = "openai"
        settings.fallback_llm_provider = "mock"
        settings.enable_budget_enforcement = False
        settings.cost_budget_cents = 100.0
        return settings
    
    @pytest.fixture
    def mock_primary_provider(self):
        """Create mock primary provider."""
        provider = AsyncMock()
        provider.name = "openai"
        provider.default_model = "gpt-3.5-turbo"
        provider.estimate_cost.return_value = 5.0
        return provider
    
    @pytest.fixture
    def mock_fallback_provider(self):
        """Create mock fallback provider."""
        provider = AsyncMock()
        provider.name = "mock"
        provider.default_model = "mock-llm"
        provider.estimate_cost.return_value = 0.0
        return provider
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample chat messages."""
        return [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="What is the capital of France?")
        ]
    
    @pytest.fixture
    def sample_llm_response(self):
        """Create sample LLM response."""
        return LLMResponse(
            content="The capital of France is Paris.",
            usage=LLMUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            model="gpt-3.5-turbo",
            finish_reason="stop",
            metadata={"estimated_cost_cents": 5.0, "provider": "openai"}
        )
    
    @patch('meulex.llm.cascade.create_llm_provider')
    def test_cascade_initialization(self, mock_create_provider, mock_settings):
        """Test LLM cascade initialization."""
        mock_primary = MagicMock()
        mock_primary.name = "openai"
        mock_fallback = MagicMock()
        mock_fallback.name = "mock"
        
        mock_create_provider.side_effect = [mock_primary, mock_fallback]
        
        cascade = LLMCascade(mock_settings)
        
        assert cascade.primary_provider == mock_primary
        assert cascade.fallback_provider == mock_fallback
        assert cascade.total_cost_cents == 0.0
        assert cascade.request_count == 0
    
    @patch('meulex.llm.cascade.create_llm_provider')
    async def test_cascade_primary_success(
        self, 
        mock_create_provider, 
        mock_settings, 
        sample_messages, 
        sample_llm_response
    ):
        """Test successful primary provider response."""
        mock_primary = AsyncMock()
        mock_primary.name = "openai"
        mock_primary.chat_completion.return_value = sample_llm_response
        mock_primary.estimate_cost = MagicMock(return_value=5.0)  # Sync method
        
        mock_create_provider.return_value = mock_primary
        
        cascade = LLMCascade(mock_settings)
        cascade.fallback_provider = None  # No fallback for this test
        
        response = await cascade.chat_completion(sample_messages)
        
        assert response == sample_llm_response
        assert cascade.total_cost_cents == 5.0
        assert cascade.request_count == 1
        mock_primary.chat_completion.assert_called_once()
    
    @patch('meulex.llm.cascade.create_llm_provider')
    async def test_cascade_fallback_on_primary_failure(
        self, 
        mock_create_provider, 
        mock_settings, 
        sample_messages, 
        sample_llm_response
    ):
        """Test fallback when primary provider fails."""
        # Setup primary provider to fail
        mock_primary = AsyncMock()
        mock_primary.name = "openai"
        mock_primary.chat_completion.side_effect = LLMError("Primary failed", "openai")
        
        # Setup fallback provider to succeed
        mock_fallback = AsyncMock()
        mock_fallback.name = "mock"
        mock_fallback.chat_completion.return_value = sample_llm_response
        mock_fallback.estimate_cost = MagicMock(return_value=0.0)  # Sync method
        
        mock_create_provider.side_effect = [mock_primary, mock_fallback]
        
        cascade = LLMCascade(mock_settings)
        
        response = await cascade.chat_completion(sample_messages)
        
        assert response == sample_llm_response
        assert cascade.total_cost_cents == 0.0  # Fallback cost
        assert cascade.request_count == 1
        
        # Verify both providers were called
        mock_primary.chat_completion.assert_called_once()
        mock_fallback.chat_completion.assert_called_once()
    
    @patch('meulex.llm.cascade.create_llm_provider')
    async def test_cascade_budget_enforcement(
        self, 
        mock_create_provider, 
        mock_settings, 
        sample_messages
    ):
        """Test budget enforcement blocks expensive requests."""
        mock_settings.enable_budget_enforcement = True
        mock_settings.cost_budget_cents = 10.0
        
        mock_primary = AsyncMock()
        mock_primary.name = "openai"
        
        mock_create_provider.return_value = mock_primary
        
        cascade = LLMCascade(mock_settings)
        cascade.total_cost_cents = 15.0  # Already over budget
        
        with pytest.raises(BudgetExceededError) as exc_info:
            await cascade.chat_completion(sample_messages)
        
        assert "Total budget exceeded" in str(exc_info.value)
        mock_primary.chat_completion.assert_not_called()
    
    @patch('meulex.llm.cascade.create_llm_provider')
    async def test_cascade_circuit_breaker_blocks_requests(
        self, 
        mock_create_provider, 
        mock_settings, 
        sample_messages
    ):
        """Test circuit breaker blocks requests when open."""
        mock_primary = AsyncMock()
        mock_primary.name = "openai"
        mock_primary.chat_completion.side_effect = LLMError("Always fails", "openai")
        
        mock_create_provider.return_value = mock_primary
        
        cascade = LLMCascade(mock_settings)
        cascade.fallback_provider = None
        
        # Trigger circuit breaker (default threshold is 5)
        for _ in range(5):
            with pytest.raises(LLMError):
                await cascade.chat_completion(sample_messages)
        
        # Circuit should be open now
        assert cascade.primary_circuit_breaker.state == "open"
        
        # Next request should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            await cascade.chat_completion(sample_messages)
    
    @patch('meulex.llm.cascade.create_llm_provider')
    def test_cascade_status_reporting(self, mock_create_provider, mock_settings):
        """Test cascade status reporting."""
        mock_primary = MagicMock()
        mock_primary.name = "openai"
        mock_primary.default_model = "gpt-3.5-turbo"
        
        mock_fallback = MagicMock()
        mock_fallback.name = "mock"
        mock_fallback.default_model = "mock-llm"
        
        mock_create_provider.side_effect = [mock_primary, mock_fallback]
        
        cascade = LLMCascade(mock_settings)
        cascade.total_cost_cents = 25.5
        cascade.request_count = 3
        
        status = cascade.get_status()
        
        assert status["primary_provider"]["name"] == "openai"
        assert status["primary_provider"]["model"] == "gpt-3.5-turbo"
        assert status["fallback_provider"]["name"] == "mock"
        assert status["budget"]["total_cost_cents"] == 25.5
        assert status["budget"]["request_count"] == 3
    
    @patch('meulex.llm.cascade.create_llm_provider')
    async def test_cascade_cleanup(self, mock_create_provider, mock_settings):
        """Test cascade cleanup closes providers."""
        mock_primary = AsyncMock()
        mock_fallback = AsyncMock()
        
        mock_create_provider.side_effect = [mock_primary, mock_fallback]
        
        cascade = LLMCascade(mock_settings)
        
        await cascade.close()
        
        mock_primary.close.assert_called_once()
        mock_fallback.close.assert_called_once()


@pytest.mark.asyncio
class TestLLMCascadeIntegration:
    """Integration tests for LLM cascade with real mock providers."""
    
    async def test_cascade_with_mock_providers(self):
        """Test cascade with actual mock providers."""
        settings = Settings(
            llm_provider="mock",
            fallback_llm_provider="mock",
            use_mock_llm=True,
            enable_budget_enforcement=False
        )
        
        cascade = LLMCascade(settings)
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Test question")
        ]
        
        response = await cascade.chat_completion(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.model == "mock-llm"
        assert response.usage.total_tokens > 0
        
        await cascade.close()
    
    async def test_cascade_budget_tracking(self):
        """Test budget tracking with mock providers."""
        settings = Settings(
            llm_provider="mock",
            use_mock_llm=True,
            enable_budget_enforcement=True,
            cost_budget_cents=0.1  # Very low budget
        )
        
        cascade = LLMCascade(settings)
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Test question")
        ]
        
        # First request should succeed (mock has 0 cost)
        response = await cascade.chat_completion(messages)
        assert response.content
        
        # Manually set high cost to test budget enforcement
        cascade.total_cost_cents = 1.0
        
        with pytest.raises(BudgetExceededError):
            await cascade.chat_completion(messages)
        
        await cascade.close()
