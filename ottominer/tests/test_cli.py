import unittest
from pathlib import Path
import tempfile
from ottominer.core.environment import env

class TestCLI(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / 'input'
        self.output_dir = self.test_dir / 'output'
        
        # Create test directories
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
    
    def test_cli_setup(self):
        """Test CLI environment setup"""
        # Test directory attributes
        self.assertTrue(hasattr(env, 'CLI_DIR'))
        self.assertTrue(hasattr(env, 'COMPLETION_DIR'))
        self.assertTrue(hasattr(env, 'CONFIG_DIR'))
        
        # Test directory existence
        self.assertTrue(env.CLI_DIR.exists())
        self.assertTrue(env.COMPLETION_DIR.exists())
        self.assertTrue(env.CONFIG_DIR.exists())
        
        # Test config file creation
        config_file = env.CONFIG_DIR / 'config.json'
        self.assertTrue(config_file.exists())
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)