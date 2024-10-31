import os
import json
from pathlib import Path
from .environment import env
from ..utils.logger import setup_logger

logger = setup_logger()

class Config:
    def __init__(self):
        self.config_file = env.CONFIG_DIR / 'config.json'
        self._config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if not self.config_file.exists():
            return self._create_default_config()
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                # Merge with default config to ensure all keys exist
                return {**self._create_default_config(), **config}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        default_config = {
            'default_output_dir': str(env.OUTPUT_DIR),
            'default_data_dir': str(env.FDATA_DIR),
            'completion_enabled': True,
            'pdf_extraction': {
                'extract_images': False,
                'dpi': 300,
                'margins': (50, 50, 0, 0),
                'table_strategy': 'lines_strict',
                'fontsize_limit': 4,
                'workers': 4,
                'output_dir': str(env.OUTPUT_DIR / 'extracted'),
                'batch_size': 100
            },
            'analysis': {
                'cache_results': True,
                'parallel_processing': True
            },
            'logging': {
                'level': 'INFO',
                'name': 'ottominer'
            }
        }
        
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error creating default config: {e}")
            
        return default_config

    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False