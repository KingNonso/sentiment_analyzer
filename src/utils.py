"""Utility functions for sentiment analysis.

This module contains helper functions for text processing, emoji handling,
and other utilities following the DRY (Don't Repeat Yourself) principle.
"""

import re
import emoji
import unicodedata
from typing import Dict, List, Tuple
from functools import lru_cache
import hashlib
import time


# Emoji sentiment mappings (simplified - in production, use a comprehensive dataset)
EMOJI_SENTIMENTS: Dict[str, float] = {
    '😀': 0.8, '😃': 0.8, '😄': 0.9, '😁': 0.8, '😆': 0.9, '😅': 0.7,
    '😂': 0.9, '🤣': 0.9, '😊': 0.7, '😇': 0.8, '🙂': 0.5, '🙃': 0.3,
    '😉': 0.6, '😌': 0.6, '😍': 0.9, '🥰': 0.9, '😘': 0.8, '😗': 0.6,
    '😙': 0.6, '😚': 0.6, '😋': 0.7, '😛': 0.6, '😜': 0.6, '🤪': 0.5,
    '😝': 0.5, '🤑': 0.4, '🤗': 0.8, '🤭': 0.4, '🤫': 0.3, '🤔': 0.0,
    '🤐': -0.2, '🤨': -0.3, '😐': 0.0, '😑': -0.2, '😶': 0.0, '😏': 0.3,
    '😒': -0.4, '🙄': -0.5, '😬': -0.2, '🤥': -0.4, '😔': -0.5, '😟': -0.6,
    '😕': -0.4, '🙁': -0.5, '☹️': -0.6, '😣': -0.6, '😖': -0.7, '😫': -0.7,
    '😩': -0.6, '🥺': -0.3, '😢': -0.7, '😭': -0.8, '😤': -0.6, '😠': -0.7,
    '😡': -0.8, '🤬': -0.9, '🤯': -0.3, '😳': -0.2, '🥵': -0.3, '🥶': -0.3,
    '😱': -0.6, '😨': -0.6, '😰': -0.6, '😥': -0.5, '😓': -0.4,
    '👍': 0.7, '👎': -0.7, '👌': 0.6, '✌️': 0.6, '🤞': 0.5, '🤟': 0.7,
    '🤘': 0.6, '🤙': 0.6, '👏': 0.7, '🙌': 0.8, '👐': 0.5, '🤲': 0.6,
    '🤝': 0.6, '🙏': 0.7, '❤️': 0.9, '🧡': 0.8, '💛': 0.8, '💚': 0.8,
    '💙': 0.8, '💜': 0.8, '🖤': -0.3, '🤍': 0.7, '🤎': 0.6, '💔': -0.8,
    '❣️': 0.8, '💕': 0.9, '💞': 0.9, '💓': 0.8, '💗': 0.8, '💖': 0.9,
    '💘': 0.8, '💝': 0.8, '💟': 0.7, '☮️': 0.6, '✨': 0.7, '🌟': 0.7,
    '⭐': 0.7, '🌈': 0.8, '🔥': 0.5, '💯': 0.8, '🎉': 0.8, '🎊': 0.8,
}


def clean_text(text: str) -> str:
    """Clean and normalize text for analysis.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    return text.strip()


def extract_emojis(text: str) -> List[str]:
    """Extract all emojis from text.
    
    Args:
        text: Input text
        
    Returns:
        List of emojis found in the text
    """
    return [char for char in text if char in emoji.EMOJI_DATA]


def calculate_emoji_sentiment(emojis: List[str]) -> Tuple[float, float]:
    """Calculate sentiment score from emojis.
    
    Args:
        emojis: List of emoji characters
        
    Returns:
        Tuple of (sentiment_score, confidence)
    """
    if not emojis:
        return 0.0, 0.0
    
    scores = []
    for emoji_char in emojis:
        if emoji_char in EMOJI_SENTIMENTS:
            scores.append(EMOJI_SENTIMENTS[emoji_char])
        else:
            # Unknown emoji - neutral
            scores.append(0.0)
    
    if not scores:
        return 0.0, 0.0
    
    avg_score = sum(scores) / len(scores)
    # Confidence based on how many emojis we recognized
    confidence = len([s for s in scores if s != 0.0]) / len(scores)
    
    return avg_score, confidence


def detect_mixed_sentiment(text: str) -> bool:
    """Detect if text contains mixed positive and negative indicators.
    
    Args:
        text: Input text
        
    Returns:
        True if mixed sentiment is detected
    """
    positive_indicators = [
        'love', 'great', 'awesome', 'excellent', 'good', 'wonderful',
        'fantastic', 'amazing', 'best', 'happy', 'joy', 'excited', 'quality'
    ]
    negative_indicators = [
        'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst',
        'disgusting', 'disappointed', 'sad', 'angry', 'frustrated', 
        'too high', 'expensive', 'price'
    ]
    
    text_lower = text.lower()
    has_positive = any(word in text_lower for word in positive_indicators)
    has_negative = any(word in text_lower for word in negative_indicators)
    
    return has_positive and has_negative


def is_emoji_only(text: str) -> bool:
    """Check if text contains only emojis and whitespace.
    
    Args:
        text: Input text
        
    Returns:
        True if text is emoji-only
    """
    # Remove all emojis and whitespace
    text_without_emoji = emoji.replace_emoji(text, '')
    text_without_emoji = text_without_emoji.strip()
    
    # If nothing left, it was emoji-only
    return len(text_without_emoji) == 0 and len(extract_emojis(text)) > 0


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length while preserving word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can preserve most of the text
        return truncated[:last_space] + '...'
    
    return truncated + '...'


def calculate_text_hash(text: str) -> str:
    """Calculate a hash of the text for caching purposes.
    
    Args:
        text: Input text
        
    Returns:
        SHA256 hash of the text
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def measure_time(func):
    """Decorator to measure function execution time.
    
    This is useful for performance monitoring and optimization.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Add execution time to result metadata if possible
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            result.metadata['execution_time_ms'] = execution_time_ms
        
        return result
    return wrapper


def normalize_score(score: float) -> float:
    """Normalize a score to the range [-1, 1].
    
    Args:
        score: Input score
        
    Returns:
        Normalized score between -1 and 1
    """
    return max(-1.0, min(1.0, score))


def get_confidence_from_score(score: float) -> float:
    """Calculate confidence based on how extreme the sentiment score is.
    
    More extreme scores (closer to -1 or 1) indicate higher confidence.
    
    Args:
        score: Sentiment score between -1 and 1
        
    Returns:
        Confidence score between 0 and 1
    """
    # Use absolute value - extreme sentiments have higher confidence
    abs_score = abs(score)
    
    if abs_score >= 0.8:
        return 0.95
    elif abs_score >= 0.6:
        return 0.85
    elif abs_score >= 0.4:
        return 0.75
    elif abs_score >= 0.2:
        return 0.65
    else:
        return 0.55  # Neutral sentiments have lower confidence


@lru_cache(maxsize=1000)
def preprocess_for_analysis(text: str) -> Dict[str, any]:
    """Preprocess text and extract features for analysis.
    
    This is cached to avoid reprocessing the same text multiple times.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary containing preprocessed text and features
    """
    cleaned = clean_text(text)
    emojis = extract_emojis(text)
    
    return {
        'original': text,
        'cleaned': cleaned,
        'emojis': emojis,
        'emoji_only': is_emoji_only(text),
        'has_mixed_sentiment': detect_mixed_sentiment(cleaned),
        'word_count': len(cleaned.split()),
        'char_count': len(text),
        'hash': calculate_text_hash(text)
    }
