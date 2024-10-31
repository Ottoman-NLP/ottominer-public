from pathlib import Path
import os
import json
from typing import Dict, Any, Optional

class Environment:
    """Environment configuration for Ottoman Miner"""
    
    def __init__(self):
        # Base directories
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.CLI_DIR = self.PROJECT_ROOT / 'cli'
        self.CORE_DIR = self.PROJECT_ROOT / 'core'
        self.FDATA_DIR = self.PROJECT_ROOT / 'fdata'
        self.OUTPUT_DIR = self.PROJECT_ROOT / 'output'
        self.CACHE_DIR = self.PROJECT_ROOT / 'cache'
        self.LOG_DIR = self.PROJECT_ROOT / 'logs'
        
        # CLI specific directories
        self.COMPLETION_DIR = Path.home() / '.ottominer' / 'completion'
        self.CONFIG_DIR = Path.home() / '.ottominer' / 'config'
        
        # Ensure critical directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for directory in [
            self.FDATA_DIR,
            self.OUTPUT_DIR,
            self.COMPLETION_DIR,
            self.CONFIG_DIR,
            self.CACHE_DIR,
            self.LOG_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_cli(self) -> bool:
        """Setup CLI environment"""
        try:
            # Ensure CLI directories exist
            self.COMPLETION_DIR.mkdir(parents=True, exist_ok=True)
            
            # Create default config if it doesn't exist
            config_file = self.CONFIG_DIR / 'config.json'
            if not config_file.exists():
                default_config = {
                    'default_output_dir': str(self.OUTPUT_DIR),
                    'default_data_dir': str(self.FDATA_DIR),
                    'completion_enabled': True
                }
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to setup CLI environment: {str(e)}")
            return False
    
    def load_json_data(self, filename: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            file_path = self.get_data_file(filename)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return {}
    
    def get_data_file(self, filename: str) -> Path:
        """Get path to data file"""
        return self.FDATA_DIR / filename
    
    def validate_path(self, path: Path) -> bool:
        """Validate path exists and is accessible"""
        try:
            if not path.exists():
                print(f"Path does not exist: {path}")
                return False
            if not path.is_dir():
                print(f"Path is not a directory: {path}")
                return False
            # Test write access
            test_file = path / '.test_write_access'
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                print(f"Path is not writable: {path} ({str(e)})")
                return False
            return True
        except Exception as e:
            print(f"Error validating path {path}: {str(e)}")
            return False

# Global environment instance
env = Environment()