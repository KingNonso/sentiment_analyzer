"""Data models for sentiment analysis.

This module contains data classes and enums used throughout the sentiment analyzer.
Following the Single Responsibility Principle, each model has one clear purpose.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class SentimentLabel(str, Enum):
    """Enumeration for sentiment classification labels.
    
    Using string enum for better JSON serialization and API compatibility.
    """
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


@dataclass(frozen=True)
class SentimentResult:
    """Immutable result of sentiment analysis.
    
    Attributes:
        score: Sentiment score between -1.0 and 1.0
        label: Categorical sentiment label
        confidence: Confidence level of the analysis (0.0 to 1.0)
        method: Method used for analysis (e.g., 'vader', 'llm')
        metadata: Additional metadata about the analysis
    
    Using frozen dataclass for immutability - results should not be modified after creation.
    """
    score: float
    label: SentimentLabel
    confidence: float
    method: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate the sentiment result values."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between -1.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass(frozen=True)
class AnalysisMetrics:
    """Metrics collected during sentiment analysis.
    
    Useful for monitoring, debugging, and performance optimization.
    """
    processing_time_ms: float
    text_length: int
    emoji_count: int
    word_count: int
    has_mixed_sentiment: bool
    language_detected: Optional[str] = None
