import logging
import sys

def setup_logger(level=logging.INFO):
    """
    Sets up a basic logger.

    Args:
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger(__name__.split('.')[0])
    logger.setLevel(level)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Initialize a default logger
logger = setup_logger() 