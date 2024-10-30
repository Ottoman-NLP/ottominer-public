import unittest
from pathlib import Path
import tempfile
import shutil
from ottominer.core.environment import env
from ottominer.core.data_manager import data_manager

class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.test_data = {
            "test_category": ["item1", "item2"]
        }
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        
    def test_environment(self):
        """Test environment setup and path management"""
        # Test critical paths exist
        self.assertTrue(env.PROJECT_ROOT.exists())
        self.assertTrue(env.CLI_DIR.exists())
        self.assertTrue(env.CORE_DIR.exists())
        self.assertTrue(env.FDATA_DIR.exists())
        self.assertTrue(env.OUTPUT_DIR.exists())
        
        # Test user-specific paths
        self.assertTrue(env.COMPLETION_DIR.exists())
        self.assertTrue(env.CONFIG_DIR.exists())
        
        # Test path relationships
        self.assertEqual(env.CLI_DIR, env.PROJECT_ROOT / 'cli')
        self.assertEqual(env.CORE_DIR, env.PROJECT_ROOT / 'core')
        self.assertEqual(env.FDATA_DIR, env.PROJECT_ROOT / 'fdata')
        self.assertEqual(env.OUTPUT_DIR, env.PROJECT_ROOT / 'output')
        
    def test_data_manager(self):
        """Test data manager functionality"""
        # Test data loading
        data_manager.load_data()
        
        # Test that we can add and retrieve markers
        test_data = {'test_marker1', 'test_marker2'}
        data_manager.add_markers('test_category', 'test_subcategory', test_data)
        
        # Verify retrieval works
        retrieved = data_manager.get_markers('test_category', 'test_subcategory')
        self.assertEqual(retrieved, test_data)