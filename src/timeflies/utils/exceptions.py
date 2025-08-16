"""Custom exceptions for TimeFlies project."""

from typing import Optional, Any


class TimeFliesError(Exception):
    """Base exception class for TimeFlies project."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} Details: {self.details}"
        return self.message


class ConfigurationError(TimeFliesError):
    """Raised when there's an error in configuration."""
    pass


class DataError(TimeFliesError):
    """Raised when there's an error with data loading or processing."""
    pass


class ModelError(TimeFliesError):
    """Raised when there's an error with model building or training."""
    pass


class PreprocessingError(TimeFliesError):
    """Raised when there's an error during data preprocessing."""
    pass


class GeneFilterError(TimeFliesError):
    """Raised when there's an error during gene filtering."""
    pass


class EvaluationError(TimeFliesError):
    """Raised when there's an error during model evaluation."""
    pass


class GPUConfigurationError(TimeFliesError):
    """Raised when there's an error configuring GPU."""
    pass


class FileNotFoundError(DataError):
    """Raised when a required file is not found."""
    pass


class InvalidParameterError(ConfigurationError):
    """Raised when an invalid parameter is provided."""
    pass


class InsufficientDataError(DataError):
    """Raised when there's insufficient data for processing."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelLoadingError(ModelError):
    """Raised when model loading fails."""
    pass


def handle_exception(func):
    """Decorator to handle exceptions and provide better error messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TimeFliesError:
            # Re-raise TimeFlies exceptions as-is
            raise
        except Exception as e:
            # Wrap other exceptions in TimeFliesError
            raise TimeFliesError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={"function": func.__name__, "args": args, "kwargs": kwargs}
            ) from e
    return wrapper