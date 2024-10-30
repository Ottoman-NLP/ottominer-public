from unittest import TestCase
from unittest.mock import patch, MagicMock
import logging
import json
from pathlib import Path
from ottominer.utils.logger import setup_logger
from ottominer.utils.decorators import handle_exceptions
from ottominer.utils.resources import check_system_resources
from ottominer.core.config import Config
import tempfile

class TestUtils(TestCase):
    def test_logger_setup(self):
        """Test logger configuration"""
        logger = setup_logger()
        
        # Test logger properties
        self.assertEqual(logger.name, 'ottominer')
        self.assertEqual(logger.level, logging.INFO)
        
        # Test handlers
        handlers = logger.handlers
        self.assertTrue(any(isinstance(h, logging.FileHandler) for h in handlers))
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in handlers))

    def test_exception_handler(self):
        """Test exception handling decorator"""
        @handle_exceptions
        def problematic_function():
            raise ValueError("Test error")
            
        # Test that exception is logged but still raised
        with self.assertRaises(ValueError):
            with self.assertLogs('ottominer', level='ERROR') as log:
                problematic_function()
                
        self.assertTrue(any("Test error" in msg for msg in log.output))

    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('os.cpu_count')
    def test_resource_checking(self, mock_cpu, mock_disk, mock_memory):
        """Test system resource checking"""
        # Mock system resources
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(available=8589934592)  # 8GB
        mock_disk.return_value = MagicMock(free=107374182400)  # 100GB
        
        resources = check_system_resources()
        
        self.assertEqual(resources['cpu_count'], 8)
        self.assertEqual(resources['memory_available'], 8589934592)
        self.assertEqual(resources['disk_free'], 107374182400)

class TestConfig(TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())  # Use tempfile instead of fixed path
        
        # Patch environment config directory
        self.patcher = patch('ottominer.core.environment.env.CONFIG_DIR', self.test_dir)
        self.mock_config_dir = self.patcher.start()
        
        self.config = Config()
    
    def test_config_creation(self):
        """Test configuration file creation and loading"""
        # Verify config file exists
        self.assertTrue(self.config.config_file.exists())
        
        # Test config structure
        self.assertIn('pdf_extraction', self.config._config)
        self.assertIn('analysis', self.config._config)
        self.assertIn('logging', self.config._config)
        
        # Test specific values
        pdf_config = self.config._config['pdf_extraction']
        self.assertEqual(pdf_config['batch_size'], 100)
        self.assertIsInstance(pdf_config['workers'], int)
    
    def tearDown(self):
        """Clean up test environment"""
        self.patcher.stop()
        import shutil
        shutil.rmtree(self.test_dir)