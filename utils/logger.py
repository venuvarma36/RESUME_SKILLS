"""
Logging utility for Resume Skill Recognition System
Provides centralized logging configuration and management.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class LoggerManager:
    """Manages logging configuration and provides logger instances."""
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls, 
                      level: str = "INFO",
                      log_to_file: bool = True,
                      log_to_console: bool = True,
                      log_dir: str = "logs",
                      log_format: str = None):
        """
        Setup logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            log_dir: Directory for log files
            log_format: Custom log format
        """
        if cls._configured:
            return
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Set log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Default format
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Create handlers
        handlers = []
        
        # Console handler with color if available
        if log_to_console:
            if COLORLOG_AVAILABLE:
                console_handler = colorlog.StreamHandler(sys.stdout)
                console_handler.setFormatter(colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S',
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                ))
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(
                    log_format,
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
            
            console_handler.setLevel(log_level)
            handlers.append(console_handler)
        
        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = log_path / f"resume_skill_recognition_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                log_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers
        )
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            Logger instance
        """
        if not cls._configured:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return LoggerManager.get_logger(name)
