"""Sentiment Analyzer Package

A sentiment analysis module for social media comments.
"""

from .analyzer import SentimentAnalyzer
from .models import SentimentResult, SentimentLabel
from .exceptions import SentimentAnalyzerError

__version__ = "1.0.0"
__all__ = ["SentimentAnalyzer", "SentimentResult", "SentimentLabel", "SentimentAnalyzerError"]
