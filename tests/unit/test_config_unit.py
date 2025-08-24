"""Tests for the generic Config class functionality."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Test the generic Config class from either project (they're identical)
from common.core.config_manager import Config


class TestConfig:
    """Test generic Config class functionality."""

    def test_config_initialization(self):
        """Test Config initialization with nested dictionaries."""
        config_dict = {
            "general": {"project_name": "test", "version": "1.0"},
            "data": {"tissue": "head", "model_type": "CNN"}
        }
        config = Config(config_dict)

        assert config.general.project_name == "test"
        assert config.data.tissue == "head"

    def test_config_attribute_access(self):
        """Test attribute access for nested configs."""
        config_dict = {"level1": {"level2": {"value": "test"}}}
        config = Config(config_dict)

        assert config.level1.level2.value == "test"

    def test_config_missing_attribute(self):
        """Test error handling for missing attributes."""
        config = Config({"existing": "value"})

        with pytest.raises(AttributeError, match="Configuration 'missing' not found"):
            _ = config.missing

    def test_config_to_dict(self):
        """Test conversion back to dictionary."""
        original_dict = {"a": {"b": "value"}, "c": "another"}
        config = Config(original_dict)
        result_dict = config.to_dict()

        assert result_dict == original_dict

    def test_config_update(self):
        """Test config update functionality."""
        config = Config({"a": 1, "b": {"c": 2}})
        update_dict = {"b": {"d": 3}, "e": 4}

        config.update(update_dict)

        assert config.a == 1
        assert config.b.c == 2
        assert config.b.d == 3
        assert config.e == 4

    def test_config_nested_access(self):
        """Test nested configuration access."""
        config_dict = {
            'general': {
                'project_name': 'TimeFlies',
                'version': '0.2.0'
            },
            'data': {
                'tissue': 'head',
                'model_type': 'CNN'
            }
        }

        config = Config(config_dict)
        assert config.general.project_name == 'TimeFlies'
        assert config.general.version == '0.2.0'
        assert config.data.tissue == 'head'
        assert config.data.model_type == 'CNN'

    def test_config_setattr(self):
        """Test setting configuration values."""
        config = Config({'test': 'value'})
        config.new_value = 'new'
        assert config.new_value == 'new'

        # Test setting nested config
        config.nested = {'key': 'value'}
        assert isinstance(config.nested, Config)
        assert config.nested.key == 'value'

    def test_config_get_method(self):
        """Test config get method with defaults."""
        config = Config({'existing': 'value'})

        assert config.get('existing') == 'value'
        assert config.get('missing') is None
        assert config.get('missing', 'default') == 'default'

        # Test nested get
        config_nested = Config({'level1': {'level2': 'value'}})
        assert config_nested.level1.get('level2') == 'value'
        assert config_nested.level1.get('missing', 'default') == 'default'

    def test_config_attribute_error(self):
        """Test AttributeError handling."""
        config = Config({})

        with pytest.raises(AttributeError, match="Configuration 'missing' not found"):
            _ = config.missing

    def test_config_repr(self):
        """Test config string representation."""
        config_dict = {'key': 'value', 'nested': {'inner': 'data'}}
        config = Config(config_dict)

        repr_str = repr(config)
        assert 'Config' in repr_str
        assert 'key' in repr_str
