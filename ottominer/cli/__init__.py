"""CLI module for Ottoman Miner"""
from ottominer.core.environment import env

# Initialize CLI environment
env.setup_cli()

from .analyzers import get_available_analyzers
from .args import parse_args, SmartArgumentParser
from .completion import generate_completion, install_completion

__all__ = [
    'parse_args',
    'SmartArgumentParser',
    'generate_completion',
    'install_completion',
    'get_available_analyzers'
]


