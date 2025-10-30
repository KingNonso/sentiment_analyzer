"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
Following the DRY principle, we centralize test setup here.
"""

import pytest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch

from src.config import Config, ThresholdConfig, LimitsConfig, LLMConfig, VADERConfig, CacheConfig
from src.analyzer import SentimentAnalyzer
from src.llm_service import MockLLMProvider


@pytest.fixture
def default_config():
    """Provide default configuration for testing."""
    return Config(
        thresholds=ThresholdConfig(
            positive_threshold=0.05,
            negative_threshold=-0.05
        ),
        limits=LimitsConfig(
            max_text_length=10000,
            min_text_length=1,
            max_emoji_only_length=20
        ),
        llm=LLMConfig(
            enabled=True,
            mock_mode=True,
            retry_attempts=2,
            retry_delay_seconds=0.1
        ),
        vader=VADERConfig(
            enabled=True,
            include_emoji=True
        ),
        enable_fallback=True,
        enable_emoji_analysis=True
    )


@pytest.fixture
def analyzer(default_config):
    """Provide a configured SentimentAnalyzer instance."""
    return SentimentAnalyzer(config=default_config)


@pytest.fixture
def mock_llm_provider():
    """Provide a mock LLM provider for testing."""
    return MockLLMProvider()


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'thresholds': {
                'positive_threshold': 0.1,
                'negative_threshold': -0.1
            },
            'limits': {
                'max_text_length': 500,
                'min_text_length': 2
            },
            'llm': {
                'enabled': False
            },
            'vader': {
                'enabled': True
            }
        }
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return {
        'positive': [
            "I absolutely love this product! It's amazing!",
            "Best purchase ever! Highly recommend 👍",
            "This made my day so much better 😊",
            "Fantastic service, couldn't be happier!"
        ],
        'negative': [
            "This is terrible, worst experience ever",
            "Completely disappointed with this 😞",
            "Waste of money, don't buy this",
            "Horrible quality, very frustrated"
        ],
        'neutral': [
            "It's okay, nothing special",
            "The product arrived today",
            "This is a comment about the item",
            "Average quality, as expected"
        ],
        'mixed': [
            "The quality is great but the price is too high",
            "Love the design but hate the color",
            "Good product, terrible customer service though"
        ],
        'emoji_only': [
            "😍😍😍",
            "😭😭😭",
            "🤔",
            "👍👍👎"
        ],
        'edge_cases': [
            "",  # Empty string
            " ",  # Whitespace only
            "a",  # Single character
            "!" * 100,  # Special characters only
            "🚀" * 100,  # Many emojis
            "This is a " + "very " * 500 + "long text",  # Very long text
        ],
        'sarcastic': [
            "Oh great, another wonderful product...",
            "Yeah right, best thing ever!!!",
            "Sure, I totally believe that /s"
        ]
    }


@pytest.fixture
def mock_external_service():
    """Mock external service calls."""
    with patch('src.llm_service.time.sleep'):  # Speed up tests
        yield


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset LRU caches before each test."""
    from src.utils import preprocess_for_analysis
    from src.config import get_default_config
    
    preprocess_for_analysis.cache_clear()
    get_default_config.cache_clear()
    
    yield


@pytest.fixture
def performance_config():
    """Configuration optimized for performance testing."""
    return Config(
        llm=LLMConfig(
            enabled=False  # Disable LLM for performance tests
        ),
        vader=VADERConfig(
            enabled=True
        ),
        cache=CacheConfig(
            enabled=True,
            max_size=1000
        )
    )
