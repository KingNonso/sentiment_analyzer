"""Tests for configuration management.

This module tests configuration loading, validation, and management.
"""

import pytest
import json
import yaml
from pathlib import Path
import tempfile
import os

from src.config import (
    Config,
    ThresholdConfig,
    LimitsConfig,
    LLMConfig,
    get_default_config
)


class TestThresholdConfig:
    """Test threshold configuration."""
    
    def test_valid_thresholds(self):
        """Test valid threshold values."""
        config = ThresholdConfig(positive_threshold=0.1, negative_threshold=-0.1)
        assert config.positive_threshold == 0.1
        assert config.negative_threshold == -0.1
    
    def test_invalid_positive_threshold(self):
        """Test invalid positive threshold."""
        with pytest.raises(ValueError, match="Positive threshold"):
            ThresholdConfig(positive_threshold=-0.5, negative_threshold=-0.1)
    
    def test_invalid_negative_threshold(self):
        """Test invalid negative threshold."""
        with pytest.raises(ValueError, match="Negative threshold"):
            ThresholdConfig(positive_threshold=0.1, negative_threshold=0.5)
    
    def test_boundary_thresholds(self):
        """Test boundary threshold values."""
        config = ThresholdConfig(positive_threshold=1.0, negative_threshold=-1.0)
        assert config.positive_threshold == 1.0
        assert config.negative_threshold == -1.0


class TestLimitsConfig:
    """Test limits configuration."""
    
    def test_default_limits(self):
        """Test default limit values."""
        config = LimitsConfig()
        assert config.max_text_length == 10000
        assert config.min_text_length == 1
        assert config.max_processing_time_ms == 5000
        assert config.max_emoji_only_length == 50
    
    def test_custom_limits(self):
        """Test custom limit values."""
        config = LimitsConfig(
            max_text_length=500,
            min_text_length=5,
            max_processing_time_ms=1000
        )
        assert config.max_text_length == 500
        assert config.min_text_length == 5
        assert config.max_processing_time_ms == 1000


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_llm_config(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.enabled is True
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.0
        assert config.max_tokens == 100
        assert config.mock_mode is True  # Default to mock mode
    
    def test_llm_config_from_env(self):
        """Test LLM configuration from environment variables."""
        os.environ["LLM_API_KEY"] = "test-key"
        os.environ["MOCK_LLM"] = "false"
        
        config = LLMConfig()
        assert config.api_key == "test-key"
        assert config.mock_mode is False
        
        # Cleanup
        del os.environ["LLM_API_KEY"]
        del os.environ["MOCK_LLM"]
    
    def test_custom_llm_config(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            enabled=False,
            model_name="gpt-4",
            temperature=0.7,
            retry_attempts=5
        )
        assert config.enabled is False
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.retry_attempts == 5


class TestMainConfig:
    """Test main configuration class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        assert config.thresholds is not None
        assert config.limits is not None
        assert config.llm is not None
        assert config.vader is not None
        assert config.cache is not None
        assert config.logging is not None
        assert config.enable_fallback is True
        assert config.enable_emoji_analysis is True
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'thresholds': {
                    'positive_threshold': 0.2,
                    'negative_threshold': -0.2
                },
                'limits': {
                    'max_text_length': 2000
                },
                'enable_fallback': False
            }
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)
        
        try:
            config = Config.from_file(temp_path)
            assert config.thresholds.positive_threshold == 0.2
            assert config.thresholds.negative_threshold == -0.2
            assert config.limits.max_text_length == 2000
            assert config.enable_fallback is False
        finally:
            temp_path.unlink()
    
    def test_config_from_json(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_content = {
                'thresholds': {
                    'positive_threshold': 0.15,
                    'negative_threshold': -0.15
                },
                'vader': {
                    'enabled': False
                }
            }
            json.dump(json_content, f)
            temp_path = Path(f.name)
        
        try:
            config = Config.from_file(temp_path)
            assert config.thresholds.positive_threshold == 0.15
            assert config.vader.enabled is False
        finally:
            temp_path.unlink()
    
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            Config.from_file(Path("nonexistent.yaml"))
    
    def test_config_unsupported_format(self):
        """Test handling of unsupported configuration format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                Config.from_file(temp_path)
        finally:
            temp_path.unlink()
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert 'thresholds' in config_dict
        assert 'limits' in config_dict
        assert 'llm' in config_dict
        assert 'vader' in config_dict
        assert 'cache' in config_dict
        assert 'logging' in config_dict
        assert 'enable_fallback' in config_dict
        assert 'enable_emoji_analysis' in config_dict
        assert 'supported_languages' in config_dict
    
    def test_config_partial_update(self):
        """Test partial configuration update from file."""
        config = Config()
        original_max_tokens = config.llm.max_tokens
        
        # Create partial config
        partial_dict = {
            'thresholds': {
                'positive_threshold': 0.3
            }
        }
        
        updated_config = Config._from_dict(partial_dict)
        
        # Should update specified values
        assert updated_config.thresholds.positive_threshold == 0.3
        
        # Should keep defaults for unspecified values
        assert updated_config.llm.max_tokens == original_max_tokens


class TestGetDefaultConfig:
    """Test default configuration retrieval."""
    
    def test_get_default_config_cached(self):
        """Test that default config is cached."""
        config1 = get_default_config()
        config2 = get_default_config()
        
        # Should return the same instance (cached)
        assert config1 is config2
    
    def test_get_default_config_no_file(self):
        """Test default config when no config file exists."""
        # Clear cache first
        get_default_config.cache_clear()
        
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.thresholds.positive_threshold == 0.05  # Default value


class TestFeatureFlags:
    """Test feature flag configuration."""
    
    def test_feature_flags_from_dict(self):
        """Test setting feature flags from dictionary."""
        config_dict = {
            'enable_fallback': False,
            'enable_emoji_analysis': False,
            'enable_language_detection': True
        }
        
        config = Config._from_dict(config_dict)
        assert config.enable_fallback is False
        assert config.enable_emoji_analysis is False
        assert config.enable_language_detection is True
    
    def test_supported_languages(self):
        """Test supported languages configuration."""
        config_dict = {
            'supported_languages': ['en', 'es', 'fr']
        }
        
        config = Config._from_dict(config_dict)
        assert 'en' in config.supported_languages
        assert 'es' in config.supported_languages
        assert 'fr' in config.supported_languages
        assert len(config.supported_languages) == 3
