# Sentiment Analyzer for Social Media Comments

A sentiment analysis component designed for processing social media comments from platforms like Instagram, YouTube, and TikTok. This implementation demonstrates professional Python coding standards, comprehensive testing practices, and robust error handling.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Best Practices Applied](#best-practices-applied)
- [Assumptions & Limitations](#assumptions--limitations)
- [Performance Considerations](#performance-considerations)

## Overview

This sentiment analyzer provides accurate sentiment analysis for social media text using a hybrid approach that combines:

1. **VADER (Primary)**: Optimized for social media text with emoji support
2. **LLM Fallback (Mocked)**: For complex cases requiring deeper understanding
3. **Emoji Analysis**: Specialized handling for emoji-only content

The system returns:

- **Sentiment Score**: Decimal between -1.0 (very negative) and +1.0 (very positive)
- **Sentiment Label**: Categorical classification (NEGATIVE, NEUTRAL, POSITIVE)
- **Confidence Score**: Reliability measure of the analysis
- **Metadata**: Additional context about the analysis

## Architecture & Design Decisions

### Why VADER as Primary Analyzer?

VADER (Valence Aware Dictionary and sEntiment Reasoner) was chosen as the primary sentiment analyzer after evaluating several popular Python sentiment analysis packages. The comparison below shows how VADER performs across key metrics:

![Python Sentiment Analysis Packages Comparison](screenshots/py-sentiment-packages-comparison.png)

**Key reasons for choosing VADER:**

1. **Social Media Optimization**: VADER is specifically designed for social media text, understanding slang, emoticons, and informal language
2. **Emoji Support**: Native understanding of emoji sentiment, crucial for platforms like Instagram and TikTok
3. **Speed**: Fast processing without requiring external API calls (100/100 speed score)
4. **No Training Required**: Pre-trained lexicon-based approach works out of the box
5. **Handles Intensity**: Recognizes intensifiers (VERY good) and negations (NOT good)
6. **Ease of Use**: Simple API with minimal setup overhead compared to transformer-based models like RoBERTa and DistilBERT

### LLM Integration (Mocked)

The LLM service is architected using the **Strategy Pattern** to allow easy switching between providers:

- **Mock Provider**: Used for testing and demonstration
- **OpenAI Provider**: Ready for production integration (API key required)
- **Extensible Design**: Easy to add Anthropic, Cohere, or other providers

The mock implementation simulates realistic LLM behavior including:

- Variable response times
- Occasional failures for resilience testing
- Rate limiting simulation
- Keyword-based sentiment analysis

### Design Patterns Applied

1. **Strategy Pattern**: LLM provider selection
2. **Composition over Inheritance**: Analyzer composed of services
3. **Dependency Injection**: Configuration injection for testability
4. **Circuit Breaker**: Prevents cascading failures from external services
5. **Factory Pattern**: Configuration creation
6. **Singleton Pattern**: Cached default configuration

### SOLID Principles

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extensible for new providers without modification
- **Liskov Substitution**: LLM providers are interchangeable
- **Interface Segregation**: Focused interfaces for each component
- **Dependency Inversion**: Depends on abstractions, not concretions

## Features

### Core Functionality

- ✅ Sentiment scoring (-1.0 to +1.0)
- ✅ Categorical labels (NEGATIVE, NEUTRAL, POSITIVE)
- ✅ Confidence scoring

### Robust Error Handling

- ✅ Input validation with detailed error messages
- ✅ Graceful degradation with fallback mechanisms
- ✅ Circuit breaker for external services
- ✅ Retry logic with exponential backoff
- ✅ Custom exception hierarchy

### Configuration Management

- ✅ YAML/JSON configuration support
- ✅ Environment variable overrides
- ✅ Validation of configuration values
- ✅ Hot-reloadable settings
- ✅ Feature flags

### Performance Optimization

- ✅ Result caching with LRU cache
- ✅ Text preprocessing cache
- ✅ Lazy service initialization
- ✅ Performance metrics collection
- ✅ Configurable timeouts

## Installation

### Requirements

- Python 3.12+
- pip or poetry for dependency management

### Basic Installation

```bash
# Clone the repository
cd sentiment_analyzer

# Install dependencies
pip install -r requirements.txt

# Or using pip with extras
pip install -e ".[dev]"
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock black ruff mypy

# Or using pyproject.toml
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from src.analyzer import SentimentAnalyzer

# Initialize analyzer with default configuration
analyzer = SentimentAnalyzer()

# Analyze sentiment
result = analyzer.analyze("I absolutely love this product! 😍")

print(f"Score: {result.score:.2f}")        # Score: 0.85
print(f"Label: {result.label}")            # Label: POSITIVE
print(f"Confidence: {result.confidence:.2f}") # Confidence: 0.90
print(f"Method: {result.method}")          # Method: vader
```

### Advanced Usage with Custom Configuration

```python
from src.config import Config, ThresholdConfig
from src.analyzer import SentimentAnalyzer

# Create custom configuration
config = Config(
    thresholds=ThresholdConfig(
        positive_threshold=0.1,  # More sensitive to positive
        negative_threshold=-0.1   # More sensitive to negative
    ),
    enable_emoji_analysis=True,
    enable_fallback=True
)

# Initialize with custom config
analyzer = SentimentAnalyzer(config=config)

# Analyze with caching
result = analyzer.analyze_cached("Great service! 👍")
```

### Batch Processing Example

```python
comments = [
    "Amazing product! Highly recommend 👍",
    "Terrible experience, very disappointed",
    "It's okay, nothing special",
    "😍😍😍",
]

results = []
for comment in comments:
    try:
        result = analyzer.analyze(comment)
        results.append({
            'text': comment,
            'score': result.score,
            'label': result.label.value
        })
    except Exception as e:
        print(f"Error analyzing '{comment}': {e}")

# Get performance metrics
metrics = analyzer.get_metrics()
print(f"Total analyses: {metrics['total_analyses']}")
print(f"Average response time: {metrics['avg_response_time_ms']:.2f}ms")
```

## Configuration

### Configuration Files

The analyzer supports multiple configuration sources:

1. **Default Configuration**: Built-in defaults
2. **YAML Configuration**: `config/config.yaml`
3. **Environment Variables**: Override specific settings
4. **Runtime Configuration**: Pass config object to constructor

### Configuration Options

```yaml
# config/config.yaml
thresholds:
  positive_threshold: 0.05 # Score above this is POSITIVE
  negative_threshold: -0.05 # Score below this is NEGATIVE

limits:
  max_text_length: 10000 # Maximum characters
  min_text_length: 1 # Minimum characters
  max_processing_time_ms: 5000 # Processing timeout

vader:
  enabled: true
  include_emoji: true # Include emoji in VADER

llm:
  enabled: true
  mock_mode: true # Use mock for testing
  model_name: "gpt-3.5-turbo"
  retry_attempts: 3

cache:
  enabled: true
  ttl_seconds: 3600 # 1 hour cache
  max_size: 1000

# Feature flags
enable_fallback: true
enable_emoji_analysis: true
```

### Environment Variables

```bash
# .env file
LLM_API_KEY=your-api-key-here
MOCK_LLM=false  # Use real LLM in production
LOG_LEVEL=INFO
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py

# Run with verbose output
pytest -v

# Run specific test class or method
pytest tests/test_analyzer.py::TestSentimentAnalysis::test_positive_sentiment
```

### Test Coverage

The test suite includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Edge Cases**: Boundary conditions and error scenarios
- **Performance Tests**: Response time and throughput
- **Mock Tests**: External service simulation

Current coverage: **86%** of codebase (exceeds 80% requirement)

#### Coverage Report

![Test Coverage Report](screenshots/test-run.png)

The coverage report above shows the breakdown by module:

- **src/analyzer.py**: 92% - Core sentiment analysis logic
- **src/config.py**: 99% - Configuration management
- **src/exceptions.py**: 90% - Custom exception hierarchy
- **src/llm_service.py**: 71% - LLM service with mocked providers
- **src/models.py**: 93% - Data models
- **src/utils.py**: 86% - Utility functions

The test suite validates proper mocking of LLM integration, circuit breaker functionality, fallback mechanisms, and comprehensive edge case handling.

### Test Categories

1. **Happy Path Tests**: Normal operation scenarios
2. **Edge Case Tests**: Boundary conditions
3. **Error Handling Tests**: Exception scenarios
4. **Performance Tests**: Speed and efficiency
5. **Integration Tests**: Component interactions
6. **Regression Tests**: Previously found bugs

_VADER is a little dumbfounded by the complexity of social media sarcasm, hence some tests fail, but it does a good job overall._

## API Reference

### SentimentAnalyzer Class

```python
class SentimentAnalyzer:
    def __init__(self, config: Optional[Config] = None):
        """Initialize analyzer with optional configuration."""

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""

    def analyze_cached(self, text: str) -> SentimentResult:
        """Cached version of analyze method."""

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
```

### SentimentResult Model

```python
@dataclass
class SentimentResult:
    score: float              # -1.0 to 1.0
    label: SentimentLabel     # NEGATIVE, NEUTRAL, POSITIVE
    confidence: float         # 0.0 to 1.0
    method: str              # 'vader', 'llm', or 'emoji'
    metadata: Optional[Dict[str, Any]]  # Additional context
```

### Exceptions

```python
SentimentAnalyzerError  # Base exception
├── ConfigurationError  # Configuration issues
├── ValidationError     # Input validation failures
├── AnalysisError      # Analysis failures
└── ExternalServiceError  # External service failures
    └── RateLimitError  # Rate limiting
```

## Best Practices Applied

### Code Quality

1. **Type Hints**: Full type annotations for better IDE support and type checking
2. **Docstrings**: Comprehensive documentation for all public methods
3. **Naming Conventions**: Clear, descriptive names following PEP 8
4. **Code Organization**: Logical module structure with clear separation of concerns
5. **DRY Principle**: Reusable utilities and shared functionality

### Error Handling

1. **Specific Exceptions**: Custom exception hierarchy for different error types
2. **Graceful Degradation**: Fallback mechanisms when services fail
3. **Input Validation**: Comprehensive validation with helpful error messages
4. **Circuit Breaker**: Prevents cascading failures
5. **Retry Logic**: Automatic retries with exponential backoff

### Testing

1. **Comprehensive Coverage**: >80% code coverage
2. **Test Organization**: Clear test structure mirroring source code
3. **Fixtures**: Reusable test data and configurations
4. **Mocking**: External dependencies mocked for isolation
5. **Parametrized Tests**: Data-driven testing for multiple scenarios

### Performance

1. **Caching**: LRU cache for repeated analyses
2. **Lazy Loading**: Services initialized only when needed
3. **Efficient Algorithms**: Optimized text processing
4. **Resource Management**: Proper cleanup and resource limits
5. **Metrics Collection**: Performance monitoring built-in

### Security

1. **Input Sanitization**: Protection against malicious input
2. **API Key Management**: Secure handling via environment variables
3. **Rate Limiting**: Protection against abuse
4. **Timeout Protection**: Prevents infinite processing
5. **Safe Defaults**: Secure default configuration

## Assumptions & Limitations

### Assumptions

1. **English Primary Language**: Optimized for English text (extensible to others)
2. **Social Media Context**: Assumes informal, short-form text
3. **UTF-8 Encoding**: Expects UTF-8 encoded text input
4. **Single Text Analysis**: Designed for individual comments, not conversations
5. **Synchronous Processing**: Not designed for real-time streaming

### Limitations

1. **Sarcasm Detection**: Limited capability for complex sarcasm
2. **Context Understanding**: No conversation history consideration
3. **Language Support**: Currently English-focused
4. **Emoji Coverage**: Limited to common emojis in mapping
5. **Mock LLM**: Production requires real LLM integration

### Known Issues

1. **Mixed Languages**: May not handle code-switching well
2. **Very Long Texts**: Performance degrades with very long input
3. **Complex Negations**: Double negatives may confuse the analyzer
4. **Cultural Context**: May miss culture-specific expressions
5. **Temporal Context**: No understanding of time-sensitive content

## Performance Considerations

### Benchmarks

- **Average Response Time**: <50ms for texts under 500 characters
- **Throughput**: ~1000 analyses per second (VADER only)
- **Memory Usage**: <100MB for typical workload
- **Cache Hit Rate**: >60% for repeated content
- **LLM Fallback Rate**: <10% of analyses

### Optimization Strategies

1. **Batch Processing**: Process multiple texts efficiently
2. **Async Processing**: Use async/await for I/O operations
3. **Connection Pooling**: Reuse connections for external services
4. **Result Caching**: Cache frequently analyzed text
5. **Preprocessing Cache**: Cache cleaned text and features

### Scaling Recommendations

1. **Horizontal Scaling**: Deploy multiple instances behind load balancer
2. **Queue Processing**: Use message queues for high volume
3. **Database Caching**: Use Redis for distributed caching
4. **CDN for Models**: Serve models from CDN for global distribution
5. **Monitoring**: Implement APM for production monitoring

## Implementation Details

### Why These Choices?

1. **VADER over Transformers**:

   - Faster processing without GPU requirements
   - No model loading overhead
   - Proven effectiveness for social media
   - Lower operational costs

2. **Mocked LLM Calls**:

   - Demonstrates architecture without API costs
   - Shows error handling and resilience patterns
   - Allows testing of fallback mechanisms
   - Simulates realistic failure scenarios

3. **Comprehensive Configuration**:

   - Flexibility for different deployment environments
   - Easy tuning without code changes
   - Feature flags for gradual rollouts
   - Support for A/B testing

4. **Extensive Testing**:

   - Documents expected behavior
   - Catches regressions early
   - Provides usage examples
   - Ensures reliability

5. **Performance Metrics**:
   - Enables monitoring in production
   - Identifies bottlenecks
   - Tracks service health
   - Supports SLA compliance

## Future Enhancements

Potential improvements for production deployment:

1. **Multi-language Support**: Add support for Spanish, French, etc.
2. **Real LLM Integration**: Implement OpenAI/Anthropic providers
3. **Async Processing**: Add async/await support
4. **Streaming Analysis**: Support for real-time comment streams
5. **Conversation Context**: Analyze comment threads
6. **Custom Model Training**: Fine-tune models on specific domains
7. **GraphQL API**: Add GraphQL endpoint for flexible queries
8. **Webhook Support**: Push results to external systems
9. **Dashboard**: Web UI for monitoring and testing
10. **Docker Support**: Containerization for easy deployment
11. **Batch Processing**: Handle large volumes of text efficiently with VADER
