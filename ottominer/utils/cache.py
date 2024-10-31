import hashlib
from pathlib import Path
from functools import wraps
import json
import pickle
from typing import Union

def cache_result(cache_dir: Path = Path(".cache")):
    """Cache extraction results."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, file_path: Union[str, Path], *args, **kwargs):
            file_path = Path(file_path)
            cache_key = hashlib.md5(str(file_path).encode()).hexdigest()
            cache_file = cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with cache_file.open('rb') as f:
                    return pickle.load(f)
            
            result = func(self, file_path, *args, **kwargs)
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_file.open('wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator 