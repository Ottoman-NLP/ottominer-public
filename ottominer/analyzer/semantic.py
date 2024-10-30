from ottominer.core import env

class SemanticAnalyzer:
    def __init__(self):
        # Load semantic labels
        self.labels = env.load_json_data('semantics.json')
        # Get cache file
        self.cache_file = env.get_cache_file('semantic_analysis') 