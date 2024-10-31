import shutil
from pathlib import Path
import tempfile
import atexit
import logging

class TempFileManager:
    """Manage temporary files with automatic cleanup."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        atexit.register(self.cleanup)
        
    def create_temp_file(self, suffix: str = None) -> Path:
        """Create a temporary file that will be cleaned up."""
        return Path(tempfile.mktemp(dir=self.temp_dir, suffix=suffix))
        
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logging.warning(f"Failed to cleanup temp files: {e}") 