import argparse
from pathlib import Path
from typing import Optional, List
from ottominer.core.environment import env
from .analyzers import get_available_analyzers

class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom formatter for better help display"""
    def __init__(self, prog):
        super().__init__(prog, max_help_position=50, width=100)

def parse_args(args=None):
    """Parse command line arguments"""
    parser = SmartArgumentParser(
        description="Ottoman Miner - Text Mining Tool for Ottoman Turkish",
        formatter_class=CustomHelpFormatter
    )

    # Add common arguments
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input directory containing PDF files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output directory for processed files'
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data processing commands')
    data_parser.add_argument(
        '--extraction-mode',
        choices=['simple', 'ocr', 'hybrid'],
        default='simple',
        help='Text extraction mode'
    )
    data_parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing'
    )
    data_parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes'
    )

    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='Text analysis commands')
    analysis_parser.add_argument(
        '--type',
        choices=get_available_analyzers(),
        default='formality',
        help='Type of analysis to perform'
    )

    return parser.parse_args(args)

class SmartArgumentParser(argparse.ArgumentParser):
    """Enhanced argument parser with better error handling"""
    
    def error(self, message: str):
        """Custom error handling with suggestions"""
        self.print_usage()
        valid_choices = self._get_valid_choices(message)
        
        if valid_choices:
            self.exit(2, f'{self.prog}: error: {message}\nDid you mean one of these?\n  ' + 
                     '\n  '.join(valid_choices) + '\n')
        else:
            self.exit(2, f'{self.prog}: error: {message}\n')

    def _get_valid_choices(self, error_message: str) -> List[str]:
        """Get valid choices for argument if available"""
        import difflib
        
        if 'invalid choice' in error_message:
            # Extract the invalid choice and valid choices from error message
            import re
            match = re.search(r"argument .+: invalid choice: '(.+)' \(choose from (.+)\)", error_message)
            if match:
                invalid_choice = match.group(1)
                valid_choices = match.group(2).replace("'", "").split(', ')
                
                # Find close matches
                return difflib.get_close_matches(invalid_choice, valid_choices, n=3, cutoff=0.6)
        
        return []

    def _check_value(self, action, value):
        """Enhanced value checking with path validation"""
        if isinstance(value, Path):
            if not env.validate_path(value):
                self.error(f"Invalid path: {value}")
        
        return super()._check_value(action, value)