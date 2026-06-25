"""
Logging configuration for Industrial Safety Monitoring System.

Provides a centralized logger with rotating file handler and colored
console output.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


# ANSI color codes for console output
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[1;31m',  # Bold Red
    'RESET': '\033[0m',
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for console output."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        color = COLORS.get(record.levelname, COLORS['RESET'])
        reset = COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)


def setup_logger(
    name: str = 'safety_monitor',
    log_file: str = 'logs/system.log',
    level: int = logging.INFO,
    max_file_size_mb: int = 5,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Args:
        name: Logger name.
        log_file: Path to the log file.
        level: Logging level (e.g., logging.INFO).
        max_file_size_mb: Maximum log file size before rotation (MB).
        backup_count: Number of backup log files to keep.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Create logs directory
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file handler: {e}")

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    try:
        console_handler.setFormatter(ColoredFormatter(log_format, date_format))
    except Exception:
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'safety_monitor') -> logging.Logger:
    """Get an existing logger or create a new one with defaults.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
