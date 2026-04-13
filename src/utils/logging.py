"""Logging utilities"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # pragma: no cover - backward compatibility
    from pythonjsonlogger import jsonlogger

    JsonFormatter = jsonlogger.JsonFormatter


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_format: str = "json",
    module_name: str = "rlaif-trading",
) -> logging.Logger:
    """Configure root logging so module-level loggers inherit handlers."""
    level = getattr(logging, log_level.upper())
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if log_format == "json":
        formatter = JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            timestamp=True,
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{module_name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger.propagate = True
    return logger


def get_logger(name: str = "rlaif-trading") -> logging.Logger:
    """Get a module logger that propagates to the configured root handlers."""
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger
