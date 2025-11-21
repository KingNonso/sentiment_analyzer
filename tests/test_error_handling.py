"""Tests for improved error handling with specific exceptions and structured logging.

This module tests that the error handling provides actionable insights
through specific exceptions and structured logging with context.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from src.llm_service import LLMService, MockLLMProvider, OpenAIProvider
from src.exceptions import (
    TextProcessingError, TextTooLongError, RateLimitError,
    LLMNotInitializedError, NoProvidersAvailableError,
    ExternalServiceError, LLMProviderError
)
from src.config import LLMConfig


class TestImprovedErrorHandling:
    """Test suite for improved error handling patterns."""
    
    def test_empty_text_validation(self):
        """Test that empty text raises specific TextProcessingError."""
        provider = MockLLMProvider()
        config = LLMConfig()
        
        # Test empty string
        with pytest.raises(TextProcessingError) as exc_info:
            provider.analyze_sentiment("", config)
        
        assert "Empty text provided for analysis" in str(exc_info.value)
        assert exc_info.value.details["provider"] == "mock-llm"
    
    def test_text_too_long_error(self):
        """Test that text exceeding max length raises TextTooLongError."""
        provider = MockLLMProvider()
        config = LLMConfig(max_tokens=10)  # Very small limit
        
        long_text = "a" * 100  # Way over the limit (10*4 = 40 chars max)
        
        with pytest.raises(TextTooLongError) as exc_info:
            provider.analyze_sentiment(long_text, config)
        
        assert exc_info.value.text_length == 100
        assert exc_info.value.max_length == 40
        assert "mock-llm" in exc_info.value.details["provider"]
    
    def test_rate_limit_error_with_context(self):
        """Test that rate limit errors include retry_after and context."""
        provider = MockLLMProvider()
        config = LLMConfig(timeout_seconds=0.01)  # Very short timeout
        
        # Consume all tokens first
        provider.request_count = 100  # Simulate many requests
        
        with patch('src.llm_service.global_rate_limiter.acquire', return_value=False):
            with patch('src.llm_service.global_rate_limiter.limiters') as mock_limiters:
                mock_limiter = Mock()
                mock_limiter.get_wait_time.return_value = 5.5
                mock_limiters.__getitem__.return_value = mock_limiter
                
                with pytest.raises(RateLimitError) as exc_info:
                    provider.analyze_sentiment("test text", config)
                
                assert exc_info.value.retry_after == 5
                assert exc_info.value.service_name == "mock-llm"
                assert "wait_time" in exc_info.value.details
                assert exc_info.value.details["wait_time"] == 5.5
                assert exc_info.value.details["request_count"] == 100
    
    def test_llm_not_initialized_error(self):
        """Test that uninitialized provider raises LLMNotInitializedError."""
        provider = OpenAIProvider(api_key=None)
        config = LLMConfig(mock_mode=False)
        
        # Patch the check so it doesn't use mock mode
        with patch.object(provider, 'api_key', None):
            with pytest.raises(LLMNotInitializedError) as exc_info:
                provider.analyze_sentiment("test text", config)
            
            assert exc_info.value.reason == "API key not provided"
            assert "OpenAI" in str(exc_info.value)
            assert exc_info.value.details["model"] == "gpt-3.5-turbo"
    
    def test_structured_logging_on_success(self, caplog):
        """Test that successful analysis logs structured data."""
        service = LLMService(LLMConfig(mock_mode=True))
        
        with caplog.at_level(logging.DEBUG):
            response = service.analyze("Happy text!")
        
        # Check for structured logging
        analysis_logs = [r for r in caplog.records if r.message == "analysis_successful"]
        assert len(analysis_logs) > 0
        
        log_record = analysis_logs[0]
        assert "provider" in log_record.__dict__
        assert "sentiment_score" in log_record.__dict__
        assert "confidence" in log_record.__dict__
        assert "response_time_ms" in log_record.__dict__
    
    def test_structured_logging_on_rate_limit(self, caplog):
        """Test that rate limit errors log structured data."""
        service = LLMService(LLMConfig(mock_mode=True))
        
        # Mock rate limit error
        with patch.object(service._get_primary_provider(), 'analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = RateLimitError(
                "Rate limited",
                service_name="test-service",
                retry_after=10,
                details={"requests_made": 100}
            )
            
            with caplog.at_level(logging.WARNING):
                try:
                    service.analyze("Test text")
                except:
                    pass
            
            # Check for structured rate limit logging
            rate_limit_logs = [r for r in caplog.records if r.message == "rate_limit_hit"]
            assert len(rate_limit_logs) > 0
            
            log_record = rate_limit_logs[0]
            assert log_record.service == "test-service"
            assert log_record.retry_after == 10
            assert log_record.details == {"requests_made": 100}
    
    def test_fallback_with_detailed_logging(self, caplog):
        """Test fallback behavior with detailed logging."""
        service = LLMService(LLMConfig(mock_mode=True))
        
        # Make primary provider fail
        primary = service._get_primary_provider()
        with patch.object(primary, 'analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = ExternalServiceError(
                "Service unavailable",
                service_name="primary",
                status_code=503
            )
            
            with caplog.at_level(logging.INFO):
                response = service.analyze("Test text")
            
            # Check for fallback logging
            fallback_logs = [r for r in caplog.records if r.message == "falling_back_to_mock"]
            assert len(fallback_logs) > 0
            
            success_logs = [r for r in caplog.records if r.message == "fallback_successful"]
            assert len(success_logs) > 0
            
            assert response.sentiment_score is not None
    
    def test_no_providers_available_error(self):
        """Test that NoProvidersAvailableError includes attempted providers."""
        service = LLMService(LLMConfig(mock_mode=True))
        
        # Make all providers fail
        for provider in service.providers.values():
            provider.is_available = Mock(return_value=False)
        
        with pytest.raises(NoProvidersAvailableError) as exc_info:
            service._get_primary_provider()
        
        assert len(exc_info.value.attempted_providers) > 0
        assert "available_providers" in exc_info.value.details
    
    def test_text_truncation_with_logging(self, caplog):
        """Test that text truncation is logged with context."""
        service = LLMService(LLMConfig(max_tokens=10))
        
        long_text = "a" * 100
        
        with caplog.at_level(logging.WARNING):
            response = service.analyze(long_text)
        
        # Check for truncation logging
        truncation_logs = [r for r in caplog.records if r.message == "text_truncated"]
        assert len(truncation_logs) > 0
        
        log_record = truncation_logs[0]
        assert log_record.original_length == 100
        assert log_record.max_length == 40  # 10 tokens * 4 chars
        assert log_record.truncated_to == 40
    
    def test_cache_error_handling(self, caplog):
        """Test that cache errors are handled gracefully."""
        service = LLMService(LLMConfig(mock_mode=True))
        
        # Mock cache to raise an error
        with patch.object(service, 'cache', new_callable=dict) as mock_cache:
            mock_cache.__contains__ = Mock(side_effect=Exception("Cache error"))
            
            with caplog.at_level(logging.WARNING):
                response = service.analyze("Test text")
            
            # Check for cache error logging
            cache_logs = [r for r in caplog.records if r.message == "cache_lookup_failed"]
            assert len(cache_logs) > 0
            
            # Should still return a valid response despite cache error
            assert response.sentiment_score is not None
    
    def test_unrecoverable_errors_not_retried(self):
        """Test that certain errors are not retried."""
        service = LLMService(LLMConfig(retry_attempts=3))
        
        # TextProcessingError should not be retried
        with patch.object(service._get_primary_provider(), 'analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = TextProcessingError("Bad text")
            
            with pytest.raises(TextProcessingError):
                service.analyze("Test")
            
            # Should only be called once (no retries)
            assert mock_analyze.call_count == 1
    
    def test_unexpected_error_wrapped(self):
        """Test that unexpected errors are wrapped in LLMProviderError."""
        service = LLMService(LLMConfig(retry_attempts=1))
        
        with patch.object(service._get_primary_provider(), 'analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = ValueError("Unexpected error")
            
            with pytest.raises(LLMProviderError) as exc_info:
                service.analyze("Test text")
            
            assert "Unexpected error during analysis" in str(exc_info.value)
            assert exc_info.value.details["original_error"] == "Unexpected error"
            assert exc_info.value.__cause__.__class__.__name__ == "ValueError"
