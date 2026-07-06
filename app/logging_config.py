# app/logging_config.py - Structured Logging Configuration
import logging
import os
import sys

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # python-json-logger < 3.0
    from pythonjsonlogger.jsonlogger import JsonFormatter


def setup_logging():
    """Setup structured logging for production"""

    class CustomJsonFormatter(JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record["service"] = "diabetes-api"
            log_record["environment"] = os.getenv("ENVIRONMENT", "development")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if os.getenv("ENVIRONMENT") == "production":
        formatter = CustomJsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
