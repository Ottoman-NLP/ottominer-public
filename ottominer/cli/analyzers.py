from typing import List

def get_available_analyzers() -> List[str]:
    """Get list of available analyzers"""
    return [
        'formality',
        'semantics',
        'genre',
        'morphology',
        'syntax'
    ]

__all__ = ['get_available_analyzers'] 