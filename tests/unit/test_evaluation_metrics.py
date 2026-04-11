"""Unit tests for evaluation and interpretation modules."""

import pytest

from timeflies.evaluation.metrics import EvaluationMetrics


@pytest.mark.unit
class TestSHAPInterpreter:
    """Test SHAP interpretation functionality."""

    def test_shap_config_validation(self, aging_config):
        """Test SHAP configuration validation."""
        assert hasattr(aging_config.interpretation, "shap")
        assert hasattr(aging_config.interpretation.shap, "enabled")
        assert hasattr(aging_config.interpretation.shap, "reference_size")

        assert isinstance(aging_config.interpretation.shap.enabled, bool)
        assert isinstance(aging_config.interpretation.shap.reference_size, int)
        assert aging_config.interpretation.shap.reference_size > 0


@pytest.mark.unit
class TestMetricsCalculator:
    """Test metrics calculator functionality."""

    def test_metrics_calculator_initialization(self, aging_config):
        """Test metrics calculator initialization."""
        calculator = EvaluationMetrics(aging_config)
        assert calculator.config == aging_config
        assert hasattr(calculator, "config")
