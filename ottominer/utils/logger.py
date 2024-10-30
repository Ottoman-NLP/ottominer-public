import logging
from pathlib import Path
from ottominer.core.environment import env

def setup_logger():
    """Setup application logging"""
    log_dir = env.PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('ottominer')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_dir / 'ottominer.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger 