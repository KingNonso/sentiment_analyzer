"""Main SentimentAnalyzer implementation.

This module implements the core sentiment analysis functionality using a hybrid approach:
1. VADER as the primary analyzer (fast, reliable for social media)
2. LLM as fallback for complex cases (mocked in this implementation)
3. Emoji analysis for social media context

Architecture decisions:
- VADER chosen for its effectiveness with social media text and emoji support
- Composition over inheritance for flexibility
- Dependency injection for testability
- Circuit breaker pattern for external service resilience
"""

import logging
import time
from typing import Optional, Dict, Any
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer # does not exist

from .models import SentimentResult, SentimentLabel
from .exceptions import (
    ValidationError, 
    AnalysisError, 
    ConfigurationError,
    ExternalServiceError
)
from .config import Config, get_default_config
from .utils import (
    calculate_emoji_sentiment,
    truncate_text,
    preprocess_for_analysis,
    normalize_score,
    get_confidence_from_score
)
from .llm_service import LLMService


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer for social media comments.
    
    This class provides sentiment analysis with:
    - Multiple analysis methods (VADER, LLM, emoji)
    - Error handling
    - Performance monitoring
    - Result caching
    - Configurable thresholds and limits
    
    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("I love this product! 😍")
        >>> print(f"Score: {result.score}, Label: {result.label}")
        Score: 0.85, Label: POSITIVE
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the sentiment analyzer.
        
        Args:
            config: Configuration object. If None, uses default configuration.
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.config = config or get_default_config()
            self._validate_config()
            
            # Initialize VADER analyzer
            self.vader = VaderAnalyzer() if self.config.vader.enabled else None
            
            # Initialize LLM service
            self.llm_service = LLMService(self.config.llm) if self.config.llm.enabled else None
            
            # Performance metrics
            self.metrics: Dict[str, Any] = {
                'total_analyses': 0,
                'vader_analyses': 0,
                'llm_analyses': 0,
                'emoji_only_analyses': 0,
                'errors': 0,
                'avg_response_time_ms': 0.0
            }
            
            # Circuit breaker for LLM service
            self.llm_failures = 0
            self.llm_circuit_open = False
            self.llm_circuit_threshold = 5
            
            logger.info("SentimentAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SentimentAnalyzer: {e}")
            raise ConfigurationError(f"Initialization failed: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if not self.config.vader.enabled and not self.config.llm.enabled:
            raise ConfigurationError("At least one analysis method must be enabled")
        
        if self.config.limits.max_text_length < self.config.limits.min_text_length:
            raise ConfigurationError("max_text_length must be >= min_text_length")
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of the given text.
        
        Args:
            text: Input text to analyze (social media comment)
            
        Returns:
            SentimentResult containing score, label, confidence, and metadata
            
        Raises:
            ValidationError: If input is invalid
            AnalysisError: If analysis fails
        """
        start_time = time.perf_counter()
        
        try:
            # Input validation
            self._validate_input(text)
            
            # Preprocess text
            processed = preprocess_for_analysis(text)
            
            # Truncate if necessary
            if processed['char_count'] > self.config.limits.max_text_length:
                text = truncate_text(text, self.config.limits.max_text_length)
                processed = preprocess_for_analysis(text)
            
            # Choose analysis method based on text characteristics
            if processed['emoji_only'] and self.config.enable_emoji_analysis:
                result = self._analyze_emoji_only(processed)
            elif self._should_use_llm(processed):
                result = self._analyze_with_llm(processed)
            else:
                result = self._analyze_with_vader(processed)
            
            # Update metrics
            self._update_metrics(result, start_time)
            
            return result
            
        except ValidationError:
            self.metrics['errors'] += 1
            raise
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Analysis failed: {e}")
            raise AnalysisError(f"Failed to analyze sentiment: {str(e)}")
    
    def _validate_input(self, text: str) -> None:
        """Validate input text.
        
        Args:
            text: Input to validate
            
        Raises:
            ValidationError: If input is invalid
        """
        if text is None:
            raise ValidationError("Input text cannot be None")
        
        if not isinstance(text, str):
            raise ValidationError(f"Input must be a string, got {type(text).__name__}")
        
        if len(text.strip()) < self.config.limits.min_text_length:
            raise ValidationError(
                f"Input text too short (min {self.config.limits.min_text_length} characters)"
            )
        
        if len(text) > self.config.limits.max_text_length * 2:  # Hard limit
            raise ValidationError(
                f"Input text too long (max {self.config.limits.max_text_length * 2} characters)"
            )
    
    def _should_use_llm(self, processed: Dict[str, Any]) -> bool:
        """Determine if LLM should be used for analysis.
        
        Args:
            processed: Preprocessed text data
            
        Returns:
            True if LLM should be used
        """
        if not self.config.llm.enabled or self.llm_circuit_open:
            return False
        
        # Use LLM for complex cases
        if processed['has_mixed_sentiment']:
            return True
        
        # Use LLM for very long texts
        if processed['word_count'] > 100:
            return True
        
        return False
    
    def _analyze_with_vader(self, processed: Dict[str, Any]) -> SentimentResult:
        """Analyze sentiment using VADER.
        
        Args:
            processed: Preprocessed text data
            
        Returns:
            SentimentResult from VADER analysis
        """
        if not self.vader:
            raise AnalysisError("VADER analyzer not initialized")
        
        logger.debug("Using VADER for sentiment analysis")
        
        # Get VADER scores
        scores = self.vader.polarity_scores(processed['cleaned'])
        compound_score = scores['compound']
        
        # Enhance with emoji sentiment if present
        if processed['emojis'] and self.config.enable_emoji_analysis:
            emoji_score, emoji_confidence = calculate_emoji_sentiment(processed['emojis'])
            if emoji_confidence > 0.5:
                # Weighted average with emoji sentiment
                compound_score = (compound_score * 0.7) + (emoji_score * 0.3)
        
        # Normalize score
        compound_score = normalize_score(compound_score)
        
        # Determine label
        label = self._score_to_label(compound_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(scores, processed)
        
        self.metrics['vader_analyses'] += 1
        
        return SentimentResult(
            score=compound_score,
            label=label,
            confidence=confidence,
            method='vader',
            metadata={
                'vader_scores': scores,
                'emoji_count': len(processed['emojis']),
                'word_count': processed['word_count'],
                'has_mixed_sentiment': processed['has_mixed_sentiment']
            }
        )
    
    def _analyze_with_llm(self, processed: Dict[str, Any]) -> SentimentResult:
        """Analyze sentiment using LLM service.
        
        Args:
            processed: Preprocessed text data
            
        Returns:
            SentimentResult from LLM analysis
        """
        if not self.llm_service:
            # Fallback to VADER
            return self._analyze_with_vader(processed)
        
        logger.debug("Using LLM for sentiment analysis")
        
        try:
            response = self.llm_service.analyze(processed['original'])
            
            # Reset circuit breaker on success
            self.llm_failures = 0
            self.llm_circuit_open = False
            
            label = self._score_to_label(response.sentiment_score)
            
            self.metrics['llm_analyses'] += 1
            
            return SentimentResult(
                score=response.sentiment_score,
                label=label,
                confidence=response.confidence,
                method='llm',
                metadata={
                    'reasoning': response.reasoning,
                    'tokens_used': response.tokens_used,
                    'response_time_ms': response.response_time_ms,
                    'emoji_count': len(processed['emojis']),
                    'word_count': processed['word_count'],
                    'has_mixed_sentiment': processed['has_mixed_sentiment']
                }
            )
            
        except ExternalServiceError as e:
            logger.warning(f"LLM service failed: {e}")
            
            # Implement circuit breaker
            self.llm_failures += 1
            if self.llm_failures >= self.llm_circuit_threshold:
                self.llm_circuit_open = True
                logger.warning("LLM circuit breaker opened")
            
            # Fallback to VADER
            if self.config.enable_fallback and self.vader:
                return self._analyze_with_vader(processed)
            
            raise AnalysisError(f"LLM analysis failed and no fallback available: {str(e)}")
    
    def _analyze_emoji_only(self, processed: Dict[str, Any]) -> SentimentResult:
        """Analyze sentiment for emoji-only text.
        
        Args:
            processed: Preprocessed text data
            
        Returns:
            SentimentResult from emoji analysis
        """
        logger.debug("Analyzing emoji-only text")
        
        emojis = processed['emojis']
        
        # Limit emoji processing
        if len(emojis) > self.config.limits.max_emoji_only_length:
            emojis = emojis[:self.config.limits.max_emoji_only_length]
        
        score, confidence = calculate_emoji_sentiment(emojis)
        label = self._score_to_label(score)
        
        self.metrics['emoji_only_analyses'] += 1
        
        return SentimentResult(
            score=score,
            label=label,
            confidence=confidence,
            method='emoji',
            metadata={
                'emoji_count': len(emojis),
                'emojis_analyzed': emojis
            }
        )
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert sentiment score to label.
        
        Args:
            score: Sentiment score between -1 and 1
            
        Returns:
            Sentiment label
        """
        if score >= self.config.thresholds.positive_threshold:
            return SentimentLabel.POSITIVE
        elif score <= self.config.thresholds.negative_threshold:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _calculate_confidence(self, vader_scores: Dict[str, float], 
                             processed: Dict[str, Any]) -> float:
        """Calculate confidence score for VADER analysis.
        
        Args:
            vader_scores: VADER sentiment scores
            processed: Preprocessed text data
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = get_confidence_from_score(vader_scores['compound'])
        
        # Adjust for mixed sentiment
        if processed['has_mixed_sentiment']:
            base_confidence *= 0.8
        
        # Adjust for text length (shorter texts have lower confidence)
        if processed['word_count'] < 5:
            base_confidence *= 0.9
        elif processed['word_count'] > 50:
            base_confidence = min(base_confidence * 1.1, 0.95)
        
        # Adjust for emoji presence
        if processed['emojis']:
            base_confidence = min(base_confidence * 1.05, 0.95)
        
        return base_confidence
    
    def _update_metrics(self, result: SentimentResult, start_time: float) -> None:
        """Update performance metrics.
        
        Args:
            result: Analysis result
            start_time: Analysis start time
        """
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        self.metrics['total_analyses'] += 1
        
        # Update average response time
        current_avg = self.metrics['avg_response_time_ms']
        total = self.metrics['total_analyses']
        self.metrics['avg_response_time_ms'] = (
            (current_avg * (total - 1) + response_time_ms) / total
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = {
            'total_analyses': 0,
            'vader_analyses': 0,
            'llm_analyses': 0,
            'emoji_only_analyses': 0,
            'errors': 0,
            'avg_response_time_ms': 0.0
        }
    
    @lru_cache(maxsize=100)
    def analyze_cached(self, text: str) -> SentimentResult:
        """Cached version of analyze method.
        
        This is useful for repeated analyses of the same text.
        Note: Only use for immutable text inputs.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Cached or newly computed SentimentResult
        """
        return self.analyze(text)
