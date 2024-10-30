import functools
from .logger import setup_logger

logger = setup_logger()

def handle_exceptions(func):
    """Decorator for consistent exception handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper 