import psutil
import os
from .logger import setup_logger

logger = setup_logger()

def check_system_resources():
    """Check available system resources"""
    memory = psutil.virtual_memory()
    cpu_count = os.cpu_count()
    disk = psutil.disk_usage('/')
    
    logger.info(f"Available CPU cores: {cpu_count}")
    logger.info(f"Available memory: {memory.available / (1024 * 1024 * 1024):.2f} GB")
    logger.info(f"Available disk space: {disk.free / (1024 * 1024 * 1024):.2f} GB")
    
    return {
        'cpu_count': cpu_count,
        'memory_available': memory.available,
        'disk_free': disk.free
    } 