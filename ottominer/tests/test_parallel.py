import pytest
from pathlib import Path
from ottominer.extractors.parallel import (
    load_stopwords,
    clean_text,
    calculate_similarity,
    extract_parallel_texts,
    ParallelPair
)

class TestParallelExtraction:
    def test_load_stopwords(self):
        """Test stopwords loading from JSON."""
        stopwords = load_stopwords()
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        assert 've' in stopwords
        assert 'ile' in stopwords

    def test_clean_text(self):
        """Test text cleaning."""
        text = "Bu, bir test metnidir! Ve bazı noktalama işaretleri içerir..."
        cleaned = clean_text(text)
        assert ',' not in cleaned
        assert '!' not in cleaned
        assert '.' not in cleaned
        assert cleaned == "bu bir test metnidir ve bazı noktalama işaretleri içerir"

    def test_calculate_similarity(self):
        """Test similarity calculation."""
        text1 = "Bu bir test metnidir"
        text2 = "Bu bir deneme yazısıdır"
        similarity = calculate_similarity(text1, text2)
        assert 0 <= similarity <= 1

    def test_extract_parallel_texts(self):
        """Test parallel text extraction."""
        text = """Original paragraph here.

_Modern translation here._

Another original.

_Another translation._"""
        
        pairs = extract_parallel_texts(text)
        assert isinstance(pairs, list)
        assert all(isinstance(p, ParallelPair) for p in pairs)

    @pytest.mark.parametrize("text1,text2,expected", [
        ("Bu bir test", "Bu bir deneme", True),
        ("", "", False),
        ("Single", "Word", False),
    ])
    def test_various_text_pairs(self, text1, text2, expected):
        """Test various text pair scenarios."""
        from ottominer.extractors.parallel import is_valid_pair
        assert is_valid_pair(text1, text2) == expected 