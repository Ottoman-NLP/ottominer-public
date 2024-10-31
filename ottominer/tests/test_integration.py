import unittest
from pathlib import Path
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
import json
import pytest
import time

from ottominer.core.environment import env
from ottominer.core.data_manager import data_manager, DataManager
from ottominer.cli.args import parse_args
from ottominer.utils.progress import ProgressTracker
from ottominer.core.config import Config
from rich.table import Table

@pytest.fixture(autouse=True)
def cleanup_progress():
    """Cleanup progress tracker before and after each test."""
    tracker = ProgressTracker()
    tracker.force_stop()
    yield
    tracker.force_stop()

class TestOttominerIntegration(unittest.TestCase):
    """Integration tests for Ottoman Miner components"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.test_dir / 'input'
        self.output_dir = self.test_dir / 'output'
        
        # Create directories
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_environment_setup(self):
        """Test environment initialization and path setup"""
        # Test project structure
        self.assertTrue(env.PROJECT_ROOT.exists())
        self.assertTrue(env.CLI_DIR.exists())
        self.assertTrue(env.CORE_DIR.exists())
        
        # Test data directories
        self.assertTrue(env.FDATA_DIR.exists())
        self.assertTrue(env.OUTPUT_DIR.exists())
        
        # Test CLI specific directories
        self.assertTrue(env.COMPLETION_DIR.exists())
        self.assertTrue(env.CONFIG_DIR.exists())
        
        # Test config file
        config_file = env.CONFIG_DIR / 'config.json'
        self.assertTrue(config_file.exists())
        
        # Test directory permissions
        self.assertTrue(os.access(env.OUTPUT_DIR, os.W_OK))
        self.assertTrue(os.access(env.FDATA_DIR, os.W_OK))

    def test_argument_validation(self):
        """Test CLI argument validation"""
        # Test valid arguments
        test_args = [
            '-i', str(self.input_dir),
            '-o', str(self.output_dir),
            'data',
            '--extraction-mode', 'simple',
            '--batch-size', '50',
            '--workers', '2'
        ]
        
        args = parse_args(test_args)
        self.assertEqual(args.input, self.input_dir)
        self.assertEqual(args.output, self.output_dir)
        
        # Test invalid arguments
        with self.assertRaises(SystemExit):
            parse_args(['-i', '/nonexistent/path'])

    def test_data_manager_integration(self):
        """Test data manager integration with environment"""
        # Create a temporary test file
        test_file = self.test_dir / 'test_data.json'
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{}')  # Start with empty JSON
        
        # Test actual data manipulation
        test_markers = {'test1', 'test2'}
        
        # Test data persistence
        data_manager.load_data(test_file)  # Load from test file
        data_manager.add_markers('test_category', 'test_subcategory', test_markers)
        
        # Reload data from file
        new_data_manager = DataManager()
        new_data_manager.load_data(test_file)
        result = new_data_manager.get_markers('test_category', 'test_subcategory')
        self.assertEqual(result, test_markers)  # Should persist between loads
        
        # Test data modification
        new_markers = {'test3', 'test4'}
        data_manager.add_markers('test_category', 'test_subcategory', new_markers)
        result = data_manager.get_markers('test_category', 'test_subcategory')
        self.assertEqual(result, test_markers | new_markers)  # Should contain all markers
        
        # Test cross-category interaction
        data_manager.add_markers('another_category', 'another_subcategory', {'test5'})
        result1 = data_manager.get_markers('test_category', 'test_subcategory')
        result2 = data_manager.get_markers('another_category', 'another_subcategory')
        self.assertNotEqual(result1, result2)  # Categories should be independent

    def test_complete_workflow(self):
        """Test complete workflow integration"""
        tracker = ProgressTracker()
        
        with tracker as progress:
            task_id = progress.add_task("Test task", total=100)
            for i in range(100):
                progress.update(task_id, advance=1)
            
            task = progress.tasks[task_id]
            assert task.completed == 100
            assert task.total == 100
        
        table = tracker.create_status_table()
        assert isinstance(table, Table)
        assert len(tracker._task_history) > 0
        
        task_info = tracker._task_history[task_id]
        assert task_info['completed'] == 100
        assert task_info['total'] == 100
        assert task_info['description'] == "Test task"

    def test_package_imports(self):
        """Test all package imports are working"""
        # Core imports
        from ottominer.core.environment import env
        from ottominer.core.data_manager import DataManager
        from ottominer.core.config import Config
        
        # CLI imports
        from ottominer.cli.args import parse_args, SmartArgumentParser
        from ottominer.cli.analyzers import get_available_analyzers
        from ottominer.cli.completion import generate_completion
        
        # Utils imports
        from ottominer.utils.logger import setup_logger
        from ottominer.utils.progress import ProgressTracker
        from ottominer.utils.resources import check_system_resources
        from ottominer.utils.decorators import handle_exceptions
        
        # Test basic functionality
        logger = setup_logger()
        config = Config()
        data_manager = DataManager()
        
        self.assertTrue(isinstance(config._config, dict))
        self.assertTrue(isinstance(get_available_analyzers(), list))
        self.assertTrue(isinstance(check_system_resources(), dict))

    def test_cross_module_integration(self):
        """Test interaction between different modules"""
        from ottominer.core.config import Config
        from ottominer.utils.logger import setup_logger
        from ottominer.core.data_manager import DataManager
        from ottominer.utils.decorators import handle_exceptions
        
        logger = setup_logger()
        config = Config()
        data_manager = DataManager()
        
        # Test decorated data manager method
        @handle_exceptions
        def test_data_operation():
            data_manager.add_markers('test', 'test_sub', {'test_marker'})
            return data_manager.get_markers('test', 'test_sub')
        
        result = test_data_operation()
        self.assertEqual(result, {'test_marker'})
        
        # Test config-based resource management
        from ottominer.utils.resources import check_system_resources
        resources = check_system_resources()
        
        # Verify both old and new config values exist
        self.assertIn('default_output_dir', config._config)
        self.assertIn('pdf_extraction', config._config)
        self.assertIn('workers', config._config['pdf_extraction'])
        
        # Test resource limits
        self.assertLessEqual(
            config._config['pdf_extraction']['workers'],
            resources['cpu_count']
        )

    def test_environment_integrity(self):
        """Test environment setup and directory structure"""
        from ottominer.core.environment import env
        
        # Test directory structure
        critical_dirs = [
            env.PROJECT_ROOT,
            env.CLI_DIR,
            env.CORE_DIR,
            env.FDATA_DIR,
            env.OUTPUT_DIR,
            env.COMPLETION_DIR,
            env.CONFIG_DIR
        ]
        
        for directory in critical_dirs:
            self.assertTrue(
                directory.exists(),
                f"Critical directory missing: {directory}"
            )
        
        # Test config file
        config_file = env.CONFIG_DIR / 'config.json'
        self.assertTrue(config_file.exists())
        
        # Test log directory
        log_dir = env.PROJECT_ROOT / 'logs'
        self.assertTrue(log_dir.exists())

    def test_progress_tracking(self):
        """Test progress tracking integration"""
        tracker = ProgressTracker()
        
        with tracker as progress:
            task_id = progress.add_task("Processing", total=10)
            for i in range(10):
                progress.update(task_id, advance=1)
                time.sleep(0.01)  # Reduced sleep time for tests
            
            task = progress.tasks[task_id]
            assert task.completed == 10
            assert task.total == 10
        
        table = tracker.create_status_table()
        assert isinstance(table, Table)
        assert len(tracker._task_history) > 0

if __name__ == '__main__':
    unittest.main(verbosity=2) 