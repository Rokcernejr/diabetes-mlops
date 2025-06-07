# app/logging_config.py - Structured Logging Configuration
import logging
import sys
import os
from pythonjsonlogger import jsonlogger

def setup_logging():
    \"\"\"Setup structured logging for production\"\"\"
    
    # Create custom formatter
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record['service'] = 'diabetes-api'
            log_record['environment'] = os.getenv('ENVIRONMENT', 'development')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Use JSON formatter for production, simple for development
    if os.getenv('ENVIRONMENT') == 'production':
        formatter = CustomJsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Usage: Add this to the top of main.py after imports:
# from app.logging_config import setup_logging
# logger = setup_logging()
