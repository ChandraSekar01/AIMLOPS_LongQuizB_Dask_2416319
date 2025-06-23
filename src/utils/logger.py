"""
Logging utility for the Dask ML Pipeline.

This module provides functions for setting up and configuring logging.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """
    Formatter for JSON-formatted logs.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)


def setup_logger(level="INFO", log_dir="logs"):
    """
    Set up and configure the logger.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir (str): Directory to save log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("dask_ml_pipeline")
    logger.setLevel(getattr(logging, level))
    logger.handlers = []  # Clear existing handlers
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        
        # Use JSON formatter for file logs
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized with level {level}")
    return logger
