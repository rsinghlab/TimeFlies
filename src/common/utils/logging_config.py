"""Logging configuration for TimeFlies project."""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any, Optional


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_dir: str | None = None,
    format_style: str = "detailed",
) -> None:
    """
    Set up logging configuration for the TimeFlies project.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional filename for log file
        log_dir: Optional directory for log files
        format_style: 'simple', 'detailed', or 'json'
    """

    # Define format styles
    formats = {
        "simple": "%(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "json": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    }

    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if log_file:
            log_file = os.path.join(log_dir, log_file)

    # Build logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": formats.get(format_style, formats["detailed"]),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            "timeflies": {"level": level, "handlers": ["console"], "propagate": False},
            "root": {"level": level, "handlers": ["console"]},
        },
    }

    # Add file handler if log file specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "standard",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }
        config["loggers"]["timeflies"]["handlers"].append("file")
        config["loggers"]["root"]["handlers"].append("file")

    # Apply configuration
    logging.config.dictConfig(config)

    # Log configuration success
    logger = logging.getLogger("timeflies.utils.logging_config")
    logger.info(f"Logging configured - Level: {level}, Format: {format_style}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a given module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"timeflies.{name}")


class LoggerMixin:
    """Mixin class to add logging functionality to other classes."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        module_name = self.__class__.__module__.replace("timeflies.", "")
        class_name = self.__class__.__name__
        return logging.getLogger(f"timeflies.{module_name}.{class_name}")


# Default configuration for development
def setup_development_logging() -> None:
    """Set up logging for development with detailed console output."""
    setup_logging(level="DEBUG", format_style="detailed")


# Default configuration for production
def setup_production_logging(log_dir: str = "logs") -> None:
    """Set up logging for production with file output."""
    setup_logging(
        level="INFO", log_file="timeflies.log", log_dir=log_dir, format_style="detailed"
    )
