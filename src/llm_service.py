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

from .exceptions import ExternalServiceError, RateLimitError
from .config import LLMConfig
from .utils import calculate_text_hash, normalize_score
from .rate_limiter import RateLimiter, global_rate_limiter

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
        
        # Proactive rate limiting - acquire token before making request
        if not global_rate_limiter.acquire(self.service_name, timeout=config.timeout_seconds):
            raise RateLimitError(
                "Rate limit would be exceeded - request throttled",
                service_name=self.service_name,
                retry_after=int(global_rate_limiter.limiters[self.service_name].get_wait_time())
            )
        
        self.request_count += 1
        self.last_request_time = time.time()
        
        # Simulate occasional failures
        if random.random() < self.failure_rate:
            raise ExternalServiceError(
                "Mock LLM service temporarily unavailable",
                service_name="MockLLM",
                status_code=503
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
            logger.info("Using mock mode for OpenAI provider")
            return self.mock_provider.analyze_sentiment(text, config)
        
        # Proactive rate limiting for actual API calls
        if not global_rate_limiter.acquire(self.service_name, timeout=config.timeout_seconds):
            raise RateLimitError(
                f"Rate limit would be exceeded for {self.service_name} - request throttled",
                service_name=self.service_name,
                retry_after=int(global_rate_limiter.limiters[self.service_name].get_wait_time())
            )
        
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
            ExternalServiceError: If all providers fail
        """
        # Check cache first
        text_hash = calculate_text_hash(text)
        if text_hash in self.cache:
            logger.debug(f"Cache hit for text hash: {text_hash}")
            return self.cache[text_hash]
        
        # Try primary provider
        primary_provider = self._get_primary_provider()
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = primary_provider.analyze_sentiment(text, self.config)
                
                # Cache the result
                self._cache_response(text_hash, response)
                
                return response
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                # With proactive rate limiting, this should rarely happen
                # But if it does, we can check if we should wait or fallback
                if e.retry_after and e.retry_after < self.config.timeout_seconds:
                    logger.info(f"Waiting {e.retry_after}s for rate limit to reset")
                    time.sleep(e.retry_after)
                elif attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    # Try fallback provider
                    return self._try_fallback(text)
                    
            except ExternalServiceError as e:
                logger.error(f"LLM service error: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    # Try fallback provider
                    return self._try_fallback(text)
        
        raise ExternalServiceError(
            "All LLM providers failed",
            service_name="LLMService"
        )
    
    def _get_primary_provider(self) -> LLMProvider:
        """Get the primary LLM provider."""
        if 'openai' in self.providers and self.providers['openai'].is_available():
            return self.providers['openai']
        return self.providers['mock']
    
    def _try_fallback(self, text: str) -> LLMResponse:
        """Try fallback provider when primary fails."""
        logger.info("Attempting fallback to mock provider")
        
        if 'mock' in self.providers:
            try:
                return self.providers['mock'].analyze_sentiment(text, self.config)
            except Exception as e:
                logger.error(f"Fallback provider also failed: {e}")
        
        raise ExternalServiceError(
            "No available LLM providers",
            service_name="LLMService"
        )
    
    def _cache_response(self, text_hash: str, response: LLMResponse) -> None:
        """Cache the LLM response with size limit."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text_hash] = response
