from pathlib import Path
from typing import Dict, Set, Optional
import json

class DataManager:
    """Manages Ottoman text analysis data"""
    
    def __init__(self):
        self._data: Dict = {}
        self._current_file: Optional[Path] = None
    
    def load_data(self, custom_path: Optional[Path] = None) -> None:
        """Load data from JSON files"""
        from ottominer.core.environment import env
        
        try:
            if custom_path:
                with open(custom_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                self._current_file = custom_path
            else:
                self._data = env.load_json_data('core_data.json')
                self._current_file = env.FDATA_DIR / 'core_data.json'
        except Exception as e:
            self._data = {}
    
    def save_data(self) -> None:
        """Save current data to file"""
        if self._current_file:
            self._current_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._current_file, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2)
    
    def get_markers(self, category: str, subcategory: str) -> Set[str]:
        """Get markers for a specific category and subcategory"""
        try:
            return set(self._data.get(category, {}).get(subcategory, []))
        except (KeyError, TypeError):
            return set()
    
    def add_markers(self, category: str, subcategory: str, markers: Set[str]) -> None:
        """Add new markers to a category"""
        if category not in self._data:
            self._data[category] = {}
        if subcategory not in self._data[category]:
            self._data[category][subcategory] = []
            
        current_markers = set(self._data[category][subcategory])
        current_markers.update(markers)
        self._data[category][subcategory] = list(current_markers)
        self.save_data()  # Auto-save when data changes



data_manager = DataManager()