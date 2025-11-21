"""LLM service for sentiment analysis with mocked API calls.

This module provides an abstraction for LLM-based sentiment analysis.
In production, this would integrate with real LLM APIs (OpenAI, Anthropic, etc.).
For this implementation, we mock the API calls to demonstrate the architecture
and error handling patterns.

Design Pattern: Strategy Pattern - allows switching between different LLM providers.
"""

import random
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .exceptions import (ExternalServiceError, RateLimitError, 
                        LLMProviderError, LLMNotInitializedError,
                        LLMTimeoutError, TextTooLongError,
                        NoProvidersAvailableError, CacheError,
                        TextProcessingError)
from .config import LLMConfig
from .utils import calculate_text_hash, normalize_score
from .rate_limiter import global_rate_limiter

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service."""
    sentiment_score: float
    confidence: float
    reasoning: Optional[str] = None
    tokens_used: int = 0
    response_time_ms: float = 0.0


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    This follows the Open/Closed Principle - open for extension, closed for modification.
    New providers can be added without modifying existing code.
    """
    
    @abstractmethod
    def analyze_sentiment(self, text: str, config: LLMConfig) -> LLMResponse:
        """Analyze sentiment using the LLM provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and demonstration.
    
    This simulates realistic LLM behavior including:
    - Variable response times
    - Occasional failures
    - Rate limiting
    - Realistic sentiment analysis based on keywords
    """
    
    def __init__(self):
        self.request_count = 0
        self.last_request_time = 0
        self.failure_rate = 0.05  # 5% failure rate for testing resilience
        self.service_name = "mock-llm"
        
        # Initialize rate limiter for this provider
        if self.service_name not in global_rate_limiter.limiters:
            global_rate_limiter.add_limiter(self.service_name, requests_per_second=10, burst_capacity=20)
        
    def analyze_sentiment(self, text: str, config: LLMConfig) -> LLMResponse:
        """Simulate LLM sentiment analysis with realistic behavior."""
        start_time = time.perf_counter()
        
        # Validate input
        if not text:
            raise TextProcessingError(
                "Empty text provided for analysis",
                details={"provider": self.service_name}
            )
        
        if len(text) > config.max_tokens * 4:  # Rough character to token ratio
            raise TextTooLongError(
                text_length=len(text),
                max_length=config.max_tokens * 4,
                details={"provider": self.service_name}
            )
        
        # Proactive rate limiting - acquire token before making request
        try:
            if not global_rate_limiter.acquire(self.service_name, timeout=config.timeout_seconds):
                wait_time = global_rate_limiter.limiters[self.service_name].get_wait_time()
                raise RateLimitError(
                    "Rate limit would be exceeded - request throttled",
                    service_name=self.service_name,
                    retry_after=int(wait_time),
                    details={
                        "wait_time": wait_time,
                        "request_count": self.request_count
                    }
                )
        except KeyError as e:
            logger.warning(
                "rate_limiter_not_configured",
                extra={"service": self.service_name, "error": str(e)}
            )
        
        self.request_count += 1
        self.last_request_time = time.time()
        
        # Simulate occasional failures
        if random.random() < self.failure_rate:
            logger.warning(
                "mock_service_failure_simulated",
                extra={
                    "service": self.service_name,
                    "request_count": self.request_count,
                    "failure_rate": self.failure_rate
                }
            )
            raise ExternalServiceError(
                "Mock LLM service temporarily unavailable",
                service_name="MockLLM",
                status_code=503,
                details={"simulated": True}
            )
        
        # Simulate processing time (20-200ms)
        time.sleep(random.uniform(0.02, 0.2))
        
        # Perform mock sentiment analysis based on keywords
        score, confidence, reasoning = self._mock_analyze(text)
        
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        return LLMResponse(
            sentiment_score=score,
            confidence=confidence,
            reasoning=reasoning,
            tokens_used=len(text.split()) * 2,  # Rough token estimate
            response_time_ms=response_time_ms
        )
    
    def _mock_analyze(self, text: str) -> Tuple[float, float, str]:
        """Perform mock sentiment analysis based on keywords and patterns."""
        text_lower = text.lower()
        
        # Sentiment keyword mappings
        strong_positive = ['amazing', 'excellent', 'fantastic', 'love', 'perfect', 'best']
        positive = ['good', 'great', 'nice', 'happy', 'like', 'enjoy', 'wonderful']
        negative = ['bad', 'poor', 'disappointing', 'hate', 'dislike', 'annoying']
        strong_negative = ['terrible', 'awful', 'horrible', 'worst', 'disgusting', 'hate']
        
        # Modifiers
        intensifiers = ['very', 'really', 'extremely', 'absolutely', 'totally']
        negations = ['not', "don't", "doesn't", "isn't", "wasn't", "never", "no"]
        
        score = 0.0
        confidence = 0.7  # Base confidence
        
        # Check for strong sentiments
        for word in strong_positive:
            if word in text_lower:
                score += 0.9
                confidence = min(confidence + 0.1, 0.95)
        
        for word in positive:
            if word in text_lower:
                score += 0.5
                confidence = min(confidence + 0.05, 0.95)
        
        for word in negative:
            if word in text_lower:
                score -= 0.5
                confidence = min(confidence + 0.05, 0.95)
        
        for word in strong_negative:
            if word in text_lower:
                score -= 0.9
                confidence = min(confidence + 0.1, 0.95)
        
        # Check for intensifiers
        for intensifier in intensifiers:
            if intensifier in text_lower:
                score *= 1.2
                confidence = min(confidence + 0.05, 0.95)
        
        # Check for negations (simple implementation)
        negation_count = sum(1 for neg in negations if neg in text_lower)
        if negation_count % 2 == 1:  # Odd number of negations
            score *= -0.8
            confidence *= 0.9  # Slightly less confident with negations
        
        # Handle sarcasm indicators (very simplified)
        if '...' in text or '!!!' in text or '/s' in text:
            score *= -0.5  # Might be sarcastic
            confidence *= 0.7
        
        # Normalize score
        score = normalize_score(score)
        
        # Generate reasoning
        if score > 0.5:
            reasoning = "Strong positive sentiment detected with high confidence indicators"
        elif score > 0:
            reasoning = "Positive sentiment detected"
        elif score < -0.5:
            reasoning = "Strong negative sentiment detected with clear negative indicators"
        elif score < 0:
            reasoning = "Negative sentiment detected"
        else:
            reasoning = "Neutral sentiment - no strong indicators found"
        
        return score, confidence, reasoning
    
    def is_available(self) -> bool:
        """Check if the mock provider is available."""
        return True  # Always available for testing


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (mocked for this implementation).
    
    In production, this would use the actual OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.mock_provider = MockLLMProvider()  # Fallback to mock
        
        # Determine service name based on model
        if "gpt-4" in model_name:
            self.service_name = "openai-gpt-4"
            requests_per_second = 1
            burst_capacity = 5
        else:
            self.service_name = "openai-gpt-3.5"
            requests_per_second = 3
            burst_capacity = 10
            
        # Initialize rate limiter for this provider
        if self.service_name not in global_rate_limiter.limiters:
            global_rate_limiter.add_limiter(self.service_name, 
                                           requests_per_second=requests_per_second,
                                           burst_capacity=burst_capacity)
        
    def analyze_sentiment(self, text: str, config: LLMConfig) -> LLMResponse:
        """Analyze sentiment using OpenAI API (mocked)."""
        if config.mock_mode or not self.api_key:
            logger.info(
                "openai_mock_mode",
                extra={
                    "model": self.model_name,
                    "has_api_key": bool(self.api_key),
                    "mock_mode": config.mock_mode
                }
            )
            return self.mock_provider.analyze_sentiment(text, config)
        
        # Check if provider is properly initialized
        if not self.api_key:
            raise LLMNotInitializedError(
                provider_name="OpenAI",
                reason="API key not provided",
                details={"model": self.model_name}
            )
        
        # Proactive rate limiting for actual API calls
        try:
            if not global_rate_limiter.acquire(self.service_name, timeout=config.timeout_seconds):
                wait_time = global_rate_limiter.limiters[self.service_name].get_wait_time()
                raise RateLimitError(
                    f"Rate limit would be exceeded for {self.service_name} - request throttled",
                    service_name=self.service_name,
                    retry_after=int(wait_time),
                    details={
                        "model": self.model_name,
                        "wait_time": wait_time
                    }
                )
        except Exception as e:
            logger.error(
                "rate_limiting_error",
                extra={
                    "service": self.service_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            # Continue without rate limiting if there's an error
        
        # In production, this would make actual API calls
        # For now, we use the mock provider after rate limiting
        return self.mock_provider.analyze_sentiment(text, config)
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return self.api_key is not None or True  # Always available in mock mode


class LLMService:
    """Service for managing LLM-based sentiment analysis.
    
    This class implements:
    - Provider selection and fallback
    - Caching of results
    - Error handling and retries
    - Performance monitoring
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.providers = self._initialize_providers()
        self.cache: Dict[str, LLMResponse] = {}
        self.max_cache_size = 100
        
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Initialize available LLM providers."""
        providers = {}
        
        if self.config.model_name.startswith('gpt'):
            providers['openai'] = OpenAIProvider(self.config.api_key, self.config.model_name)
        
        # Always include mock provider as fallback
        providers['mock'] = MockLLMProvider()
        
        # Configure default rate limits for all services
        global_rate_limiter.configure_default_limits()
        
        return providers
    
    def analyze(self, text: str) -> LLMResponse:
        """Analyze sentiment using configured LLM provider with fallback.
        
        Args:
            text: Text to analyze
            
        Returns:
            LLMResponse with sentiment analysis results
            
        Raises:
            NoProvidersAvailableError: If all providers fail
            TextTooLongError: If text exceeds maximum length
        """
        # Validate input
        if not text or not text.strip():
            raise TextProcessingError(
                "Empty or whitespace-only text provided",
                details={"text_length": len(text) if text else 0}
            )
        
        # Check text length
        max_chars = self.config.max_tokens * 4  # Rough estimate
        if len(text) > max_chars:
            logger.warning(
                "text_truncated",
                extra={
                    "original_length": len(text),
                    "max_length": max_chars,
                    "truncated_to": max_chars
                }
            )
            text = text[:max_chars]
        
        # Check cache first
        text_hash = calculate_text_hash(text)
        try:
            if text_hash in self.cache:
                logger.debug(
                    "cache_hit",
                    extra={
                        "text_hash": text_hash,
                        "cache_size": len(self.cache)
                    }
                )
                return self.cache[text_hash]
        except Exception as e:
            logger.warning(
                "cache_lookup_failed",
                extra={"error": str(e), "text_hash": text_hash}
            )
        
        # Try primary provider
        primary_provider = self._get_primary_provider()
        attempted_providers = []
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    "attempting_analysis",
                    extra={
                        "provider": primary_provider.__class__.__name__,
                        "attempt": attempt + 1,
                        "text_length": len(text)
                    }
                )
                
                response = primary_provider.analyze_sentiment(text, self.config)
                
                # Cache the result
                try:
                    self._cache_response(text_hash, response)
                except Exception as e:
                    logger.warning(
                        "cache_write_failed",
                        extra={"error": str(e), "text_hash": text_hash}
                    )
                
                logger.info(
                    "analysis_successful",
                    extra={
                        "provider": primary_provider.__class__.__name__,
                        "sentiment_score": response.sentiment_score,
                        "confidence": response.confidence,
                        "response_time_ms": response.response_time_ms
                    }
                )
                
                return response
                
            except RateLimitError as e:
                logger.warning(
                    "rate_limit_hit",
                    extra={
                        "service": e.service_name,
                        "retry_after": e.retry_after,
                        "attempt": attempt + 1,
                        "details": e.details
                    }
                )
                attempted_providers.append(e.service_name)
                
                # With proactive rate limiting, this should rarely happen
                # But if it does, we can check if we should wait or fallback
                if e.retry_after and e.retry_after < self.config.timeout_seconds:
                    logger.info(
                        "waiting_for_rate_limit",
                        extra={"wait_time": e.retry_after}
                    )
                    time.sleep(e.retry_after)
                elif attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay_seconds * (attempt + 1)
                    logger.debug(
                        "retry_delay",
                        extra={"wait_time": wait_time, "attempt": attempt + 1}
                    )
                    time.sleep(wait_time)
                else:
                    # Try fallback provider
                    return self._try_fallback(text, attempted_providers)
                    
            except LLMTimeoutError as e:
                logger.error(
                    "llm_timeout",
                    extra={
                        "provider": e.service_name,
                        "timeout": e.timeout_seconds,
                        "attempt": attempt + 1
                    }
                )
                attempted_providers.append(e.service_name)
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    return self._try_fallback(text, attempted_providers)
                    
            except (LLMNotInitializedError, TextProcessingError) as e:
                # These are unrecoverable errors - don't retry
                logger.error(
                    "unrecoverable_error",
                    extra={
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "details": getattr(e, 'details', {})
                    }
                )
                raise
                
            except ExternalServiceError as e:
                logger.error(
                    "llm_service_error",
                    extra={
                        "service": e.service_name,
                        "status_code": e.status_code,
                        "attempt": attempt + 1,
                        "details": e.details
                    }
                )
                attempted_providers.append(e.service_name)
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    # Try fallback provider
                    return self._try_fallback(text, attempted_providers)
                    
            except Exception as e:
                logger.exception(
                    "unexpected_llm_error",
                    extra={
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
                )
                
                if attempt >= self.config.retry_attempts - 1:
                    raise LLMProviderError(
                        f"Unexpected error during analysis: {e}",
                        service_name=primary_provider.__class__.__name__,
                        details={"original_error": str(e)}
                    ) from e
        
        raise ExternalServiceError(
            "All LLM providers failed",
            service_name="LLMService"
        )
    
    def _get_primary_provider(self) -> LLMProvider:
        """Get the primary LLM provider."""
        # For now, return the first available provider
        for provider in self.providers.values():
            if provider.is_available():
                return provider
        raise NoProvidersAvailableError(
            attempted_providers=list(self.providers.keys()),
            details={"available_providers": list(self.providers.keys())}
        )
    
    def _try_fallback(self, text: str, attempted_providers: list[str] | None = None) -> LLMResponse:
        """Try fallback provider when primary fails.
        
        Args:
            text: Text to analyze
            attempted_providers: List of providers that have already been attempted
            
        Returns:
            LLMResponse from fallback provider
            
        Raises:
            NoProvidersAvailableError: If all fallback attempts fail
        """
        attempted_providers = attempted_providers or []
        
        if 'mock' in self.providers and 'mock' not in attempted_providers:
            logger.info(
                "falling_back_to_mock",
                extra={
                    "attempted_providers": attempted_providers,
                    "text_length": len(text)
                }
            )
            try:
                response = self.providers['mock'].analyze_sentiment(text, self.config)
                logger.info(
                    "fallback_successful",
                    extra={
                        "provider": "mock",
                        "sentiment_score": response.sentiment_score
                    }
                )
                return response
            except Exception as e:
                logger.error(
                    "fallback_provider_failed",
                    extra={
                        "provider": "mock",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                attempted_providers.append('mock')
        
        # No more providers available
        raise NoProvidersAvailableError(
            attempted_providers=attempted_providers,
            details={
                "text_length": len(text),
                "available_providers": list(self.providers.keys())
            }
        )
    
    def _cache_response(self, text_hash: str, response: LLMResponse) -> None:
        """Cache the LLM response with size limit."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text_hash] = response
