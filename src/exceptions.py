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


class LLMProviderError(ExternalServiceError):
    """Base exception for LLM provider-specific errors."""
    pass


class LLMNotInitializedError(LLMProviderError):
    """Raised when LLM provider is not properly initialized."""
    def __init__(self, 
                 provider_name: str,
                 reason: str,
                 details: Optional[dict[str, Any]] = None) -> None:
        message = f"LLM provider '{provider_name}' not initialized: {reason}"
        super().__init__(message, provider_name, details=details)
        self.reason = reason


class LLMTimeoutError(LLMProviderError):
    """Raised when LLM request times out."""
    def __init__(self, 
                 provider_name: str,
                 timeout_seconds: float,
                 details: Optional[dict[str, Any]] = None) -> None:
        message = f"LLM request timed out after {timeout_seconds}s"
        super().__init__(message, provider_name, status_code=408, details=details)
        self.timeout_seconds = timeout_seconds


class TextProcessingError(AnalysisError):
    """Raised when text preprocessing fails."""
    pass


class TextTooLongError(ValidationError):
    """Raised when input text exceeds maximum length."""
    def __init__(self, 
                 text_length: int,
                 max_length: int,
                 details: Optional[dict[str, Any]] = None) -> None:
        message = f"Text length ({text_length}) exceeds maximum ({max_length})"
        super().__init__(message, details=details)
        self.text_length = text_length
        self.max_length = max_length


class NoProvidersAvailableError(LLMProviderError):
    """Raised when no LLM providers are available."""
    def __init__(self, 
                 attempted_providers: list[str],
                 details: Optional[dict[str, Any]] = None) -> None:
        message = f"No LLM providers available. Attempted: {', '.join(attempted_providers)}"
        super().__init__(message, "all_providers", details=details)
        self.attempted_providers = attempted_providers


class CacheError(SentimentAnalyzerError):
    """Raised when cache operations fail."""
    pass
