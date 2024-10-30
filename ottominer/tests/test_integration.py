import unittest
from pathlib import Path
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
import json

from ottominer.core.environment import env
from ottominer.core.data_manager import data_manager, DataManager
from ottominer.cli.args import parse_args
from ottominer.utils.progress import ProgressTracker

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

    @patch('ottominer.utils.progress.Live')
    def test_progress_tracking(self, mock_live):
        """Test progress tracker integration"""
        progress = ProgressTracker()
        
        # Test progress tracking
        progress.start("Test Operation", 3)
        
        # Simulate file processing
        test_files = ['file1.pdf', 'file2.pdf', 'file3.pdf']
        for file in test_files:
            progress.update(file, {
                "Pages": "5/5",
                "Memory": "100MB"
            })
            
        self.assertEqual(progress.processed_files, 3)
        self.assertEqual(progress.total_files, 3)
        
        progress.stop()

    def test_argument_environment_integration(self):
        """Test integration between CLI arguments and environment"""
        test_args = [
            '-i', str(self.input_dir),
            '-o', str(self.output_dir),
            'data',
            '--extraction-mode', 'simple'
        ]
        
        args = parse_args(test_args)
        
        # Verify environment can handle the paths
        self.assertTrue(env.validate_path(args.input))
        self.assertTrue(env.validate_path(args.output))

    def test_complete_workflow(self):
        """Test complete workflow integration"""
        with patch('ottominer.utils.progress.Live') as mock_live:
            # Create test input file
            test_content = "This is a test document with some markers."
            test_file = self.input_dir / "test.txt"
            test_file.write_text(test_content)

            # Create a mock result
            mock_result = {
                'formality': {
                    'matches': ['test', 'document'],
                    'stats': {'total': 2}
                }
            }
            
            # Write mock result to simulate analysis
            output_file = self.output_dir / "test_analysis.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mock_result, f)

            # Test argument parsing
            args = parse_args([
                '-i', str(self.input_dir),
                '-o', str(self.output_dir),
                'analysis',
                '--type', 'formality'
            ])
            
            # Verify arguments are correct
            self.assertEqual(str(args.input), str(self.input_dir))
            self.assertEqual(str(args.output), str(self.output_dir))
            self.assertEqual(args.type, 'formality')
            
            # Verify directory structure
            self.assertTrue(self.input_dir.exists())
            self.assertTrue(self.output_dir.exists())
            self.assertTrue(test_file.exists())
            self.assertTrue(output_file.exists())
            
            # Verify output format
            with open(output_file) as f:
                results = json.load(f)
            self.assertIn('formality', results)
            self.assertIn('matches', results['formality'])
            self.assertIn('stats', results['formality'])

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

if __name__ == '__main__':
    unittest.main(verbosity=2) 