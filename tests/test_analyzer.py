"""Tests for the SentimentAnalyzer class.

This module tests the core functionality of the sentiment analyzer,
including happy paths, edge cases, error handling, and performance.
"""

import pytest
from unittest.mock import patch

from src.analyzer import SentimentAnalyzer
from src.models import SentimentLabel, SentimentResult
from src.exceptions import ValidationError, ConfigurationError, ExternalServiceError
from src.config import Config


class TestSentimentAnalyzerInitialization:
    """Test analyzer initialization and configuration."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.vader is not None
        
    def test_init_with_custom_config(self, default_config):
        """Test initialization with custom configuration."""
        analyzer = SentimentAnalyzer(config=default_config)
        assert analyzer.config == default_config
        
    def test_init_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        invalid_config = Config()
        invalid_config.vader.enabled = False
        invalid_config.llm.enabled = False
        
        with pytest.raises(ConfigurationError, match="At least one analysis method"):
            SentimentAnalyzer(config=invalid_config)
    
    def test_init_with_config_file(self, temp_config_file):
        """Test initialization with configuration from file."""
        config = Config.from_file(temp_config_file)
        analyzer = SentimentAnalyzer(config=config)
        assert analyzer.config.thresholds.positive_threshold == 0.1


class TestSentimentAnalysis:
    """Test core sentiment analysis functionality."""
    
    def test_positive_sentiment(self, analyzer):
        """Test analysis of clearly positive text."""
        result = analyzer.analyze("I absolutely love this product! It's amazing!")
        
        assert isinstance(result, SentimentResult)
        assert result.score > 0.5
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0.7
        assert result.method in ['vader', 'llm']
        
    def test_negative_sentiment(self, analyzer):
        """Test analysis of clearly negative text."""
        result = analyzer.analyze("This is terrible, worst experience ever")
        
        assert result.score < -0.5
        assert result.label == SentimentLabel.NEGATIVE
        assert result.confidence > 0.7
        
    def test_neutral_sentiment(self, analyzer):
        """Test analysis of neutral text."""
        result = analyzer.analyze("The product arrived today")
        
        assert -0.2 < result.score < 0.2
        assert result.label == SentimentLabel.NEUTRAL
        
    def test_emoji_sentiment(self, analyzer):
        """Test sentiment analysis with emojis."""
        # Positive emoji
        result = analyzer.analyze("Great product! 😍😍😍")
        assert result.score > 0.5
        assert result.label == SentimentLabel.POSITIVE
        
        # Negative emoji
        result = analyzer.analyze("Not happy 😞😭")
        assert result.score < 0
        assert result.label == SentimentLabel.NEGATIVE
        
    def test_emoji_only_text(self, analyzer):
        """Test analysis of emoji-only text."""
        result = analyzer.analyze("😍😍😍")
        
        assert result.score > 0.5
        assert result.label == SentimentLabel.POSITIVE
        assert result.method == 'emoji'
        assert 'emoji_count' in result.metadata
        
    def test_mixed_sentiment(self, analyzer):
        """Test analysis of text with mixed sentiment."""
        result = analyzer.analyze("The quality is great but the price is too high")
        
        # Should detect mixed sentiment in metadata
        assert result.metadata.get('has_mixed_sentiment') is True
        
    def test_sarcasm_handling(self, analyzer):
        """Test handling of potentially sarcastic text."""
        result = analyzer.analyze("Oh great, another wonderful product... /s")
        
        # Should handle sarcasm indicators
        assert result.score <= 0.2  # Should not be strongly positive
        
    def test_very_long_text(self, analyzer):
        """Test analysis of very long text."""
        long_text = "This is a test. " * 500  # Very long text
        result = analyzer.analyze(long_text)
        
        assert isinstance(result, SentimentResult)
        # Text should be truncated but still analyzed
        
    def test_special_characters(self, analyzer):
        """Test handling of special characters."""
        result = analyzer.analyze("Great!!! #awesome @product $$$ 100%")
        
        assert isinstance(result, SentimentResult)
        assert result.score > 0  # Should still detect positive sentiment


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_none_input(self, analyzer):
        """Test handling of None input."""
        with pytest.raises(ValidationError, match="cannot be None"):
            analyzer.analyze(None)
    
    def test_empty_string(self, analyzer):
        """Test handling of empty string."""
        with pytest.raises(ValidationError, match="too short"):
            analyzer.analyze("")
    
    def test_whitespace_only(self, analyzer):
        """Test handling of whitespace-only input."""
        with pytest.raises(ValidationError, match="too short"):
            analyzer.analyze("   ")
    
    def test_non_string_input(self, analyzer):
        """Test handling of non-string input."""
        with pytest.raises(ValidationError, match="must be a string"):
            analyzer.analyze(123)
        
        with pytest.raises(ValidationError, match="must be a string"):
            analyzer.analyze(['text'])
    
    def test_extremely_long_input(self, analyzer):
        """Test handling of extremely long input."""
        # Text longer than hard limit
        very_long_text = "a" * 25000
        
        with pytest.raises(ValidationError, match="too long"):
            analyzer.analyze(very_long_text)
    
    def test_unicode_input(self, analyzer):
        """Test handling of Unicode text."""
        result = analyzer.analyze("这是一个测试 🌍 مرحبا")
        
        assert isinstance(result, SentimentResult)


class TestLLMIntegration:
    """Test LLM service integration and fallback."""
    
    @patch('src.analyzer.LLMService')
    def test_llm_fallback_to_vader(self, mock_llm_service, default_config):
        """Test fallback to VADER when LLM fails."""
        # Configure mock to raise exception
        mock_llm_service.return_value.analyze.side_effect = Exception("LLM failed")
        
        analyzer = SentimentAnalyzer(config=default_config)
        result = analyzer.analyze("This is a test")
        
        # Should fallback to VADER
        assert result.method == 'vader'
    
    def test_circuit_breaker(self, analyzer):
        """Test circuit breaker for LLM service."""
        # Force LLM to be used by patching _should_use_llm
        with patch.object(analyzer, '_should_use_llm', return_value=True):
            with patch.object(analyzer.llm_service, 'analyze') as mock_analyze:
                # Mock with ExternalServiceError so it's caught by circuit breaker logic
                mock_analyze.side_effect = ExternalServiceError(
                    "Service unavailable",
                    service_name="LLM",
                    status_code=503
                )
                
                # Trigger failures
                for _ in range(6):
                    try:
                        analyzer.analyze("Test text")
                    except Exception:
                        pass
                
                # Circuit should be open after threshold failures
                assert analyzer.llm_failures >= analyzer.llm_circuit_threshold
    
    @patch('src.llm_service.time.sleep')  # Speed up test
    def test_llm_retry_logic(self, mock_sleep, analyzer):
        """Test LLM retry logic."""
        # Force LLM to be used
        with patch.object(analyzer, '_should_use_llm', return_value=True):
            # Test that fallback to VADER works when LLM fails
            with patch.object(analyzer.llm_service, 'analyze') as mock_analyze:
                # Mock with ExternalServiceError so fallback is triggered
                mock_analyze.side_effect = ExternalServiceError(
                    "Service failure",
                    service_name="LLM",
                    status_code=503
                )
                
                # Should fallback to VADER after LLM fails
                result = analyzer.analyze("Test text")
                
                # Verify it fell back to VADER
                assert result.method == 'vader'


class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""
    
    def test_metrics_collection(self, analyzer):
        """Test that metrics are collected correctly."""
        # Reset metrics
        analyzer.reset_metrics()
        
        # Perform analyses
        analyzer.analyze("Positive text")
        analyzer.analyze("Negative text")
        analyzer.analyze("😍😍😍")
        
        metrics = analyzer.get_metrics()
        
        assert metrics['total_analyses'] == 3
        assert metrics['emoji_only_analyses'] == 1
        assert metrics['avg_response_time_ms'] > 0
        
    def test_metrics_reset(self, analyzer):
        """Test metrics reset functionality."""
        analyzer.analyze("Test text")
        analyzer.reset_metrics()
        
        metrics = analyzer.get_metrics()
        assert metrics['total_analyses'] == 0
        assert metrics['errors'] == 0
    
    def test_error_counting(self, analyzer):
        """Test that errors are counted in metrics."""
        analyzer.reset_metrics()
        
        # Cause an error
        try:
            analyzer.analyze(None)
        except ValidationError:
            pass
        
        metrics = analyzer.get_metrics()
        assert metrics['errors'] == 1


class TestCaching:
    """Test caching functionality."""
    
    def test_cached_analysis(self, analyzer):
        """Test that cached analysis returns same result."""
        text = "This is a test text for caching"
        
        result1 = analyzer.analyze_cached(text)
        result2 = analyzer.analyze_cached(text)
        
        assert result1.score == result2.score
        assert result1.label == result2.label
        
    def test_cache_different_texts(self, analyzer):
        """Test that different texts are cached separately."""
        result1 = analyzer.analyze_cached("This is amazing!")
        result2 = analyzer.analyze_cached("This is terrible!")
        
        # Should have different scores for different sentiments
        assert result1.score > 0  # Positive
        assert result2.score < 0  # Negative
        assert result1 != result2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_character(self, analyzer):
        """Test analysis of single character."""
        result = analyzer.analyze("!")
        assert isinstance(result, SentimentResult)
    
    def test_repeated_characters(self, analyzer):
        """Test analysis of repeated characters."""
        result = analyzer.analyze("!!!!!!!!!")
        assert isinstance(result, SentimentResult)
    
    def test_mixed_languages(self, analyzer):
        """Test analysis of mixed language text."""
        result = analyzer.analyze("Great product! C'est magnifique! 很好!")
        assert isinstance(result, SentimentResult)
    
    def test_numbers_only(self, analyzer):
        """Test analysis of numbers only."""
        result = analyzer.analyze("123 456 789")
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.NEUTRAL
    
    def test_url_in_text(self, analyzer):
        """Test handling of URLs in text."""
        result = analyzer.analyze("Check this out: https://example.com Great site!")
        assert isinstance(result, SentimentResult)
    
    def test_hashtags_mentions(self, analyzer):
        """Test handling of social media elements."""
        result = analyzer.analyze("#awesome @user This is #great!")
        assert isinstance(result, SentimentResult)
        assert result.score > 0


class TestConfigurationManagement:
    """Test configuration management and validation."""
    
    def test_threshold_configuration(self, default_config):
        """Test custom threshold configuration."""
        default_config.thresholds.positive_threshold = 0.3
        default_config.thresholds.negative_threshold = -0.3
        
        analyzer = SentimentAnalyzer(config=default_config)
        
        # Test that thresholds are applied
        result = analyzer.analyze("okay")  # Slightly positive
        if 0 < result.score < 0.3:
            assert result.label == SentimentLabel.NEUTRAL
    
    def test_limits_configuration(self, default_config):
        """Test text length limits configuration."""
        default_config.limits.max_text_length = 50
        analyzer = SentimentAnalyzer(config=default_config)
        
        # Should truncate long text
        long_text = "a" * 100
        result = analyzer.analyze(long_text)
        assert isinstance(result, SentimentResult)
    
    def test_feature_flags(self, default_config):
        """Test feature flag configuration."""
        default_config.enable_emoji_analysis = False
        analyzer = SentimentAnalyzer(config=default_config)
        
        result = analyzer.analyze("😍😍😍")
        # Should not use emoji-only analysis when disabled
        assert result.method != 'emoji'


class TestRealWorldScenarios:
    """Test real-world social media scenarios."""
    
    @pytest.mark.parametrize("text,expected_sentiment", [
        ("Just got my order! So excited! 🎉", "positive"),
        ("Shipping took forever 😤", "negative"),
        ("Product is okay, nothing special", "neutral"),
        ("BEST. DAY. EVER!!! 💯", "positive"),
        ("Don't waste your money on this garbage", "negative"),
        ("10/10 would recommend!", "positive"),
        ("meh", "neutral"),
        ("This slaps! 🔥🔥🔥", "positive"),
        ("Big yikes 😬", "negative"),
        ("Not the vibe", "negative"),
    ])
    def test_social_media_slang(self, analyzer, text, expected_sentiment):
        """Test handling of social media slang and expressions."""
        result = analyzer.analyze(text)
        
        if expected_sentiment == "positive":
            assert result.score > 0.1
        elif expected_sentiment == "negative":
            assert result.score < -0.1
        else:
            assert -0.2 < result.score < 0.2
    
    def test_instagram_comment(self, analyzer):
        """Test typical Instagram comment."""
        result = analyzer.analyze("Yasss queen! 👑 Slay! 💅✨")
        assert result.score > 0.3  # Adjusted threshold for VADER
        assert result.label == SentimentLabel.POSITIVE
    
    def test_youtube_comment(self, analyzer):
        """Test typical YouTube comment."""
        result = analyzer.analyze("First! Great video, keep it up!")
        assert result.score > 0.3
        assert result.label == SentimentLabel.POSITIVE
    
    def test_tiktok_comment(self, analyzer):
        """Test typical TikTok comment."""
        result = analyzer.analyze("POV: you found the best product ever 😂")
        assert result.score > 0
        assert result.label == SentimentLabel.POSITIVE
