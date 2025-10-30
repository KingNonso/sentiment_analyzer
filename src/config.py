"""Configuration management for the sentiment analyzer.

Implements configuration using:
- Pydantic for validation and type safety
- Environment variables for sensitive data
- YAML for default configurations
- Override hierarchy: ENV > config file > defaults

This follows the Dependency Inversion Principle - high-level modules
depend on abstractions (Config class) not concrete implementations.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json
import yaml
from functools import lru_cache


@dataclass
class ThresholdConfig:
    """Thresholds for sentiment classification.
    
    These values determine how scores map to labels.
    """
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05
    
    def __post_init__(self) -> None:
        """Validate threshold values."""
        if not -1.0 <= self.negative_threshold <= 0.0:
            raise ValueError("Negative threshold must be between -1.0 and 0.0")
        if not 0.0 <= self.positive_threshold <= 1.0:
            raise ValueError("Positive threshold must be between 0.0 and 1.0")


@dataclass
class LimitsConfig:
    """Input limits and constraints."""
    max_text_length: int = 10000  # Characters
    max_processing_time_ms: int = 5000  # Milliseconds
    min_text_length: int = 1
    max_emoji_only_length: int = 50  # Max emojis to process if text is emoji-only


@dataclass
class LLMConfig:
    """Configuration for LLM-based analysis (mocked in this implementation)."""
    enabled: bool = True
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0  # Use 0 for consistent results
    max_tokens: int = 100
    timeout_seconds: int = 10
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    mock_mode: bool = field(default_factory=lambda: os.getenv("MOCK_LLM", "true").lower() == "true")


@dataclass
class VADERConfig:
    """Configuration for VADER sentiment analysis."""
    enabled: bool = True
    include_emoji: bool = True
    boost_negation: bool = True
    alpha: float = 0.75  # Normalization alpha parameter


@dataclass
class CacheConfig:
    """Configuration for result caching."""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour
    max_size: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_metrics: bool = True


@dataclass
class Config:
    """Main configuration class for the sentiment analyzer.
    
    This class aggregates all configuration sections and provides
    a single point of access for configuration throughout the application.
    """
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vader: VADERConfig = field(default_factory=VADERConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_fallback: bool = True
    enable_emoji_analysis: bool = True
    enable_language_detection: bool = False  # Can be enabled if needed
    
    # Supported languages (for future multi-language support)
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML or JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Config instance with loaded values
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        
        if 'thresholds' in data:
            config.thresholds = ThresholdConfig(**data['thresholds'])
        if 'limits' in data:
            config.limits = LimitsConfig(**data['limits'])
        if 'llm' in data:
            config.llm = LLMConfig(**data['llm'])
        if 'vader' in data:
            config.vader = VADERConfig(**data['vader'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        # Update feature flags
        for key in ['enable_fallback', 'enable_emoji_analysis', 'enable_language_detection']:
            if key in data:
                setattr(config, key, data[key])
        
        if 'supported_languages' in data:
            config.supported_languages = data['supported_languages']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (useful for serialization)."""
        return {
            'thresholds': {
                'positive_threshold': self.thresholds.positive_threshold,
                'negative_threshold': self.thresholds.negative_threshold,
            },
            'limits': {
                'max_text_length': self.limits.max_text_length,
                'max_processing_time_ms': self.limits.max_processing_time_ms,
                'min_text_length': self.limits.min_text_length,
                'max_emoji_only_length': self.limits.max_emoji_only_length,
            },
            'llm': {
                'enabled': self.llm.enabled,
                'model_name': self.llm.model_name,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'timeout_seconds': self.llm.timeout_seconds,
                'retry_attempts': self.llm.retry_attempts,
                'retry_delay_seconds': self.llm.retry_delay_seconds,
                'mock_mode': self.llm.mock_mode,
            },
            'vader': {
                'enabled': self.vader.enabled,
                'include_emoji': self.vader.include_emoji,
                'boost_negation': self.vader.boost_negation,
                'alpha': self.vader.alpha,
            },
            'cache': {
                'enabled': self.cache.enabled,
                'ttl_seconds': self.cache.ttl_seconds,
                'max_size': self.cache.max_size,
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'enable_metrics': self.logging.enable_metrics,
            },
            'enable_fallback': self.enable_fallback,
            'enable_emoji_analysis': self.enable_emoji_analysis,
            'enable_language_detection': self.enable_language_detection,
            'supported_languages': self.supported_languages,
        }


@lru_cache(maxsize=1)
def get_default_config() -> Config:
    """Get the default configuration instance.
    
    This is cached to ensure we only create one instance.
    """
    # Check for config file in standard locations
    config_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path.home() / ".sentiment_analyzer" / "config.yaml",
    ]
    
    for path in config_paths:
        if path.exists():
            return Config.from_file(path)
    
    # Return default config if no file found
    return Config()
