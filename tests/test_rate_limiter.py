"""Tests for the rate limiting module.

This module tests the token bucket rate limiter implementation to ensure
it properly controls request rates and handles burst capacity.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest

from src.rate_limiter import RateLimiter, MultiServiceRateLimiter


class TestRateLimiter:
    """Test suite for the RateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization with different parameters."""
        # Default burst capacity
        limiter = RateLimiter(requests_per_second=5)
        assert limiter.rate == 5
        assert limiter.capacity == 5
        assert limiter.tokens == 5
        
        # Custom burst capacity
        limiter = RateLimiter(requests_per_second=5, burst_capacity=10)
        assert limiter.rate == 5
        assert limiter.capacity == 10
        assert limiter.tokens == 10
    
    def test_acquire_single_token(self):
        """Test acquiring a single token."""
        limiter = RateLimiter(requests_per_second=10)
        
        # Should be able to acquire immediately
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.tokens < 10
    
    def test_acquire_multiple_tokens_burst(self):
        """Test acquiring multiple tokens within burst capacity."""
        limiter = RateLimiter(requests_per_second=5, burst_capacity=10)
        
        # Should be able to acquire 10 tokens immediately (burst)
        acquired = 0
        for _ in range(10):
            if limiter.try_acquire():
                acquired += 1
        
        assert acquired == 10
        assert limiter.tokens < 0.01  # Allow for tiny timing differences
        
        # Next acquire should fail without waiting
        assert limiter.try_acquire() is False
    
    def test_token_refill(self):
        """Test that tokens are refilled over time."""
        limiter = RateLimiter(requests_per_second=10)
        
        # Consume all tokens
        for _ in range(10):
            limiter.try_acquire()
        
        assert limiter.tokens < 0.01  # Allow for tiny timing differences
        
        # Wait for tokens to refill (should get ~5 tokens after 0.5 seconds)
        time.sleep(0.5)
        
        # Check tokens have been refilled
        available = limiter.get_available_tokens()
        assert 4 <= available <= 6  # Allow some timing variance
    
    def test_rate_limiting_enforcement(self):
        """Test that rate limiting is properly enforced."""
        limiter = RateLimiter(requests_per_second=10, burst_capacity=5)
        
        # Track request times
        request_times = []
        
        # Make 10 requests
        for _ in range(10):
            if limiter.acquire(timeout=2):
                request_times.append(time.time())
        
        # First 5 should be immediate (burst), next 5 should be rate limited
        assert len(request_times) == 10
        
        # Check that requests after burst are properly spaced
        for i in range(6, 10):
            time_diff = request_times[i] - request_times[i-1]
            # Should be approximately 0.1 seconds apart (10 req/s = 0.1s per req)
            assert 0.08 <= time_diff <= 0.12  # Allow some variance
    
    def test_timeout_behavior(self):
        """Test timeout behavior when tokens are not available."""
        limiter = RateLimiter(requests_per_second=1, burst_capacity=1)
        
        # Consume the single token
        assert limiter.acquire(timeout=0.1) is True
        
        # Next acquire should timeout
        start = time.time()
        assert limiter.acquire(timeout=0.1) is False
        elapsed = time.time() - start
        
        # Check if timeout occurred (should be very quick)
        assert elapsed < 0.02  # Should fail almost immediately
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access to rate limiter."""
        limiter = RateLimiter(requests_per_second=10, burst_capacity=20)
        successful_acquires = []
        lock = threading.Lock()
        
        def try_acquire():
            if limiter.try_acquire():
                with lock:
                    successful_acquires.append(time.time())
                return True
            return False
        
        # Launch multiple threads trying to acquire tokens
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(try_acquire) for _ in range(30)]
            results = [f.result() for f in as_completed(futures)]
        
        # Should have exactly 20 successful acquires (burst capacity)
        assert len(successful_acquires) == 20
        assert results.count(True) == 20
        assert results.count(False) == 10


class TestMultiServiceRateLimiter:
    """Test suite for the MultiServiceRateLimiter class."""
    
    def test_initialization(self):
        """Test multi-service rate limiter initialization."""
        multi_limiter = MultiServiceRateLimiter()
        assert multi_limiter.limiters == {}
    
    def test_add_limiter(self):
        """Test adding rate limiters for different services."""
        multi_limiter = MultiServiceRateLimiter()
        
        multi_limiter.add_limiter("service1", requests_per_second=5)
        multi_limiter.add_limiter("service2", requests_per_second=10, burst_capacity=20)
        
        assert "service1" in multi_limiter.limiters
        assert "service2" in multi_limiter.limiters
        assert multi_limiter.limiters["service1"].rate == 5
        assert multi_limiter.limiters["service2"].capacity == 20
    
    def test_acquire_from_different_services(self):
        """Test acquiring tokens from different services."""
        multi_limiter = MultiServiceRateLimiter()
        multi_limiter.add_limiter("fast", requests_per_second=100)
        multi_limiter.add_limiter("slow", requests_per_second=1)
        
        # Fast service should allow immediate acquisition
        assert multi_limiter.acquire("fast", timeout=0.1) is True
        
        # Slow service should allow first acquisition
        assert multi_limiter.acquire("slow", timeout=0.1) is True
        
        # Second slow acquisition should fail quickly
        assert multi_limiter.acquire("slow", timeout=0.1) is False
        
        # Fast service should still work
        assert multi_limiter.acquire("fast", timeout=0.1) is True
    
    def test_unknown_service(self):
        """Test behavior with unknown service."""
        multi_limiter = MultiServiceRateLimiter()
        
        # Should return True for unknown service (no limiting)
        assert multi_limiter.acquire("unknown", timeout=0.1) is True
        assert multi_limiter.try_acquire("unknown") is True
    
    def test_configure_default_limits(self):
        """Test configuring default rate limits."""
        multi_limiter = MultiServiceRateLimiter()
        multi_limiter.configure_default_limits()
        
        # Check that default services are configured
        assert "openai-gpt-3.5" in multi_limiter.limiters
        assert "openai-gpt-4" in multi_limiter.limiters
        assert "anthropic-claude" in multi_limiter.limiters
        assert "youtube-api" in multi_limiter.limiters
        assert "mock-llm" in multi_limiter.limiters
        
        # Check specific rate limits
        assert multi_limiter.limiters["openai-gpt-3.5"].rate == 3
        assert multi_limiter.limiters["openai-gpt-4"].rate == 1
        assert multi_limiter.limiters["youtube-api"].rate == 100
    
    def test_service_isolation(self):
        """Test that rate limits are isolated between services."""
        multi_limiter = MultiServiceRateLimiter()
        multi_limiter.add_limiter("service1", requests_per_second=2, burst_capacity=2)
        multi_limiter.add_limiter("service2", requests_per_second=2, burst_capacity=2)
        
        # Consume all tokens from service1
        assert multi_limiter.acquire("service1", timeout=0.1) is True
        assert multi_limiter.acquire("service1", timeout=0.1) is True
        assert multi_limiter.try_acquire("service1") is False
        
        # Service2 should still have tokens available
        assert multi_limiter.acquire("service2", timeout=0.1) is True
        assert multi_limiter.acquire("service2", timeout=0.1) is True
        assert multi_limiter.try_acquire("service2") is False


class TestRateLimiterIntegration:
    """Integration tests for rate limiter with realistic scenarios."""
    
    def test_api_simulation(self):
        """Simulate API calls with rate limiting."""
        multi_limiter = MultiServiceRateLimiter()
        multi_limiter.add_limiter("api", requests_per_second=5, burst_capacity=10)
        
        # Track successful API calls
        successful_calls = []
        failed_calls = []
        
        # Simulate 20 API call attempts
        for i in range(20):
            if multi_limiter.try_acquire("api"):
                successful_calls.append((i, time.time()))
            else:
                failed_calls.append((i, time.time()))
        
        # First 10 should succeed (burst), next 10 should fail
        assert len(successful_calls) == 10
        assert len(failed_calls) == 10
        
        # Wait for tokens to refill
        time.sleep(1)
        
        # Should be able to make 5 more calls
        additional_calls = 0
        for _ in range(10):
            if multi_limiter.try_acquire("api"):
                additional_calls += 1
        
        assert 4 <= additional_calls <= 6  # Allow timing variance
    
    def test_multiple_apis_different_rates(self):
        """Test managing multiple APIs with different rate limits."""
        multi_limiter = MultiServiceRateLimiter()
        
        # Configure different APIs with realistic limits
        multi_limiter.add_limiter("openai", requests_per_second=3, burst_capacity=5)
        multi_limiter.add_limiter("youtube", requests_per_second=100, burst_capacity=200)
        multi_limiter.add_limiter("slow_api", requests_per_second=0.5, burst_capacity=1)
        
        # Test that each API respects its own limits
        openai_success = sum(1 for _ in range(10) if multi_limiter.try_acquire("openai"))
        youtube_success = sum(1 for _ in range(250) if multi_limiter.try_acquire("youtube"))
        slow_success = sum(1 for _ in range(5) if multi_limiter.try_acquire("slow_api"))
        
        assert openai_success == 5  # Burst capacity
        assert youtube_success == 200  # Burst capacity
        assert slow_success == 1  # Burst capacity
