from typing import Dict, Set, List, Union, Optional
from dataclasses import dataclass

@dataclass
class DataSchema:
    """Schema definition for Ottoman Miner data files"""
    
    categories: Dict[str, Set[str]] = None
    
    def __post_init__(self):
        self.categories = {
            'formality': {'formal_markers', 'informal_markers'},
            'semantics': {'religious', 'cultural', 'political', 'economic'},
            'genre': {'persian_compounds', 'arabic_patterns', 'honorifics'},
            'stopwords': {'particles_and_conjunctions'},
            'suffixes': {'case_markers', 'possessive', 'verbal'}
        }
    
    def validate_data(self, data: Dict) -> bool:
        """Validate data structure against schema"""
        for category, subcategories in self.categories.items():
            if category not in data:
                return False
            for subcategory in subcategories:
                if subcategory not in data[category]:
                    return False
        return True

__all__ = ['DataSchema']