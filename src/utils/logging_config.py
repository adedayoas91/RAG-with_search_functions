"""
Centralized logging configuration for Agentic RAG System.

This module provides logging setup with file rotation, colored console output,
and performance timing decorators.
"""

import logging
import functools
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Any
from datetime import datetime


# ANSI color codes for console output
class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        # Format the message
        result = super().format(record)

        # Reset level name to avoid affecting other handlers
        record.levelname = levelname

        return result


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "./data/agentic_rag.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure centralized logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Create data directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = ColorFormatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function

    Example:
        @log_performance
        def slow_function():
            time.sleep(1)
            return "done"
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            logger.debug(f"Starting {func.__name__}")
            result = wrapper(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Error in {func.__name__} after {elapsed:.2f}s: {str(e)}",
                exc_info=True
            )
            raise

    return wrapper


def log_api_call(
    provider: str,
    endpoint: str,
    method: str = "POST",
    **kwargs: Any
) -> None:
    """
    Log API call details for debugging and cost tracking.

    Args:
        provider: API provider (e.g., 'OpenAI', 'Tavily')
        endpoint: API endpoint or operation
        method: HTTP method
        **kwargs: Additional metadata to log
    """
    logger = get_logger("api_calls")
    metadata = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"[{provider}] {method} {endpoint} | {metadata}")


def log_exception(
    exception: Exception,
    context: str = "",
    **kwargs: Any
) -> None:
    """
    Log exception with full context and stack trace.

    Args:
        exception: Exception instance
        context: Contextual information about where error occurred
        **kwargs: Additional metadata
    """
    logger = get_logger("exceptions")
    metadata = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    error_msg = f"Exception in {context}: {str(exception)}"
    if metadata:
        error_msg += f" | {metadata}"
    logger.error(error_msg, exc_info=True)


def log_session_start(session_id: str, query: str) -> None:
    """
    Log start of a new RAG session.

    Args:
        session_id: Unique session identifier
        query: User query
    """
    logger = get_logger("session")
    logger.info(f"=== Session {session_id} Started ===")
    logger.info(f"Query: {query}")


def log_session_end(
    session_id: str,
    duration: float,
    total_cost: float,
    sources_processed: int
) -> None:
    """
    Log end of RAG session with summary statistics.

    Args:
        session_id: Unique session identifier
        duration: Session duration in seconds
        total_cost: Total cost in USD
        sources_processed: Number of sources processed
    """
    logger = get_logger("session")
    logger.info(f"=== Session {session_id} Completed ===")
    logger.info(
        f"Duration: {duration:.2f}s | "
        f"Cost: ${total_cost:.4f} | "
        f"Sources: {sources_processed}"
    )


# Initialize logging on module import
def _initialize_default_logging() -> None:
    """Initialize with default settings if not already configured."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()


# Auto-initialize with defaults
_initialize_default_logging()
