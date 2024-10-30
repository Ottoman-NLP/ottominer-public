import unittest
from pathlib import Path
import tempfile
import json
from ottominer.core.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_manager = DataManager()
        
        # Create a simple test data structure - content doesn't matter
        self.test_data = {
            'test_category': {
                'test_subcategory': ['test_item']
            }
        }
        
        # Save test data
        test_file = self.test_dir / 'test_data.json'
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f)
            
    def test_load_data(self):
        """Test data loading functionality"""
        # Test loading invalid data
        invalid_path = self.test_dir / "nonexistent.json"
        self.data_manager.load_data(invalid_path)
        self.assertEqual(self.data_manager._data, {})  # Should handle missing files
        
        # Test loading malformed data
        bad_json = self.test_dir / "bad.json"
        bad_json.write_text("{not valid json")
        self.data_manager.load_data(bad_json)
        self.assertEqual(self.data_manager._data, {})  # Should handle invalid JSON
        
        # Test data operations after loading
        markers = {'test1', 'test2'}
        self.data_manager.add_markers('category', 'subcategory', markers)
        result = self.data_manager.get_markers('category', 'subcategory')
        self.assertEqual(result, markers)
        
        # Test data type conversion
        self.data_manager.add_markers('category', 'subcategory', ['test3'])  # Add as list
        result = self.data_manager.get_markers('category', 'subcategory')
        self.assertTrue(isinstance(result, set))  # Should always return sets
        