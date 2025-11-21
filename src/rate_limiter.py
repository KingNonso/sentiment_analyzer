"""Token bucket rate limiter for API calls.

This module provides a proactive rate limiting strategy using the token bucket algorithm.
Instead of reactively handling rate limit errors and retrying, this approach prevents
hitting rate limits by controlling the request rate upfront.

Design Benefits:
- Prevents wasted compute on failed requests
- Provides predictable API call patterns
- Reduces latency from retry delays
- Better resource utilization
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for controlling API request rates.

    The token bucket algorithm allows for burst capacity while maintaining
    an average rate limit. Tokens are added to the bucket at a constant rate,
    and each request consumes one token.

    Attributes:
        rate: Maximum requests per second
        capacity: Maximum bucket capacity (allows for bursts)
        tokens: Current number of tokens in the bucket
        last_update: Timestamp of last token update
    """
    
    def __init__(self, requests_per_second: float, burst_capacity: int | None = None):
        """Initialize the rate limiter.

        Args:
            requests_per_second: Maximum sustained request rate
            burst_capacity: Maximum burst size (defaults to rate)
        """
        self.rate = requests_per_second
        self.capacity = burst_capacity or requests_per_second
        self.tokens = self.capacity
        self.last_update = time.time()
        self.lock = threading.Lock()
        
    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire permission to make a request.

        This method blocks until a token is available or the timeout is reached.

        Args:
            timeout: Maximum time to wait for a token (None = wait forever)

        Returns:
            True if token was acquired, False if timeout reached
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                self._refill_tokens()
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    logger.debug(f"Token acquired. Remaining: {self.tokens:.2f}")
                    return True
                
                # Calculate wait time for next token
                wait_time = (1 - self.tokens) / self.rate
                
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    logger.warning(f"Rate limiter timeout after {elapsed:.2f}s")
                    return False

            # Wait for next token
            time.sleep(min(wait_time, 0.1))  # Check every 100ms max

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token was available, False otherwise
        """
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= 1:
                self.tokens -= 1
                logger.debug(f"Token acquired (non-blocking). Remaining: {self.tokens:.2f}")
                return True
            return False
            
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now
        
    def get_wait_time(self) -> float:
        """Get the estimated wait time for the next available token.
        
        Returns:
            Estimated seconds until next token is available
        """
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= 1:
                return 0.0
            return (1 - self.tokens) / self.rate
            
    def get_available_tokens(self) -> float:
        """Get the current number of available tokens.
        
        Returns:
            Number of tokens currently available
        """
        with self.lock:
            self._refill_tokens()
            return self.tokens


class MultiServiceRateLimiter:
    """Manages rate limiters for multiple API services.

    This class provides a centralized way to manage rate limiting across
    different API providers, each with their own rate limits.
    """

    def __init__(self):
        """Initialize the multi-service rate limiter."""
        self.limiters: dict[str, RateLimiter] = {}
        self.lock = threading.Lock()
        
    def add_limiter(self, service_name: str, requests_per_second: float,
                   burst_capacity: int | None = None) -> None:
        """Add a rate limiter for a specific service.

        Args:
            service_name: Name of the service
            requests_per_second: Rate limit for the service
            burst_capacity: Burst capacity for the service
        """
        with self.lock:
            self.limiters[service_name] = RateLimiter(requests_per_second, burst_capacity)
            logger.info(f"Added rate limiter for {service_name}: {requests_per_second} req/s")
            
    def acquire(self, service_name: str, timeout: float | None = None) -> bool:
        """Acquire a token for a specific service.

        Args:
            service_name: Name of the service
            timeout: Maximum time to wait for a token

        Returns:
            True if token was acquired, False if timeout or service not found
        """
        limiter = self.limiters.get(service_name)
        if limiter:
            return limiter.acquire(timeout)
        logger.warning(f"No rate limiter found for service: {service_name}")
        return True  # Allow request if no limiter is configured
        
    def try_acquire(self, service_name: str) -> bool:
        """Try to acquire a token without blocking.

        Args:
            service_name: Name of the service

        Returns:
            True if token was available, False otherwise
        """
        limiter = self.limiters.get(service_name)
        if limiter:
            return limiter.try_acquire()
        return True  # Allow request if no limiter is configured
        
    def configure_default_limits(self) -> None:
        """Configure default rate limits for common services."""
        # OpenAI limits (varies by tier, using conservative values)
        self.add_limiter("openai-gpt-3.5", requests_per_second=3, burst_capacity=10)
        self.add_limiter("openai-gpt-4", requests_per_second=1, burst_capacity=5)
        
        # Anthropic Claude limits
        self.add_limiter("anthropic-claude", requests_per_second=2, burst_capacity=5)
        
        # Google/YouTube API limits
        self.add_limiter("youtube-api", requests_per_second=100, burst_capacity=200)
        
        # Mock provider (for testing)
        self.add_limiter("mock-llm", requests_per_second=10, burst_capacity=20)
        
        logger.info("Default rate limits configured")


# Global rate limiter instance
global_rate_limiter = MultiServiceRateLimiter()
