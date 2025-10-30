"""Custom exceptions for the sentiment analyzer.

Following best practices for exception handling:
- Specific exceptions for different error scenarios
- Preserving original exception context
- Providing useful error messages
"""

from typing import Optional, Any


class SentimentAnalyzerError(Exception):
    """Base exception for all sentiment analyzer errors.
    
    All custom exceptions inherit from this base class, allowing
    consumers to catch all analyzer-specific errors with a single except clause.
    """
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(SentimentAnalyzerError):
    """Raised when there's an issue with configuration."""
    pass


class ValidationError(SentimentAnalyzerError):
    """Raised when input validation fails."""
    pass


class AnalysisError(SentimentAnalyzerError):
    """Raised when sentiment analysis fails."""
    pass


class ExternalServiceError(SentimentAnalyzerError):
    """Raised when an external service (API, model) fails."""
    def __init__(self, 
                 message: str, 
                 service_name: str,
                 status_code: Optional[int] = None,
                 details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, details)
        self.service_name = service_name
        self.status_code = status_code


class RateLimitError(ExternalServiceError):
    """Raised when rate limit is exceeded."""
    def __init__(self, 
                 message: str,
                 service_name: str,
                 retry_after: Optional[int] = None,
                 details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, service_name, status_code=429, details=details)
        self.retry_after = retry_after
