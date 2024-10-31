from typing import List, Dict, Set
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class ParallelPair:
    """Data class for parallel text pairs."""
    original: str
    translation: str
    similarity: float
    pair_type: str
    metadata: Dict = None

def load_stopwords() -> Set[str]:
    """Load stopwords from JSON file."""
    try:
        stopwords_path = Path(__file__).parent.parent / 'fdata' / 'stopwords.json'
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('particles_and_conjunctions', []))
    except Exception as e:
        logger.error(f"Error loading stopwords: {e}")
        return set()

# Load stopwords from JSON file
OTTOMAN_STOPWORDS = load_stopwords()

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove special characters but keep Turkish characters
    text = re.sub(r'[^a-zA-ZğĞıİöÖşŞüÜçÇ\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.lower()

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    try:
        # Clean and normalize texts
        text1 = clean_text(text1)
        text2 = clean_text(text2)
        
        if not text1 or not text2:
            return 0.0
        
        # Create word sets for basic overlap calculation
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
            
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def is_valid_pair(original: str, translation: str, min_words: int = 2, 
                 min_similarity: float = 0.2) -> bool:
    """Validate if the pair is legitimate."""
    if not original or not translation:
        return False
        
    # Basic validation
    original_words = original.split()
    translation_words = translation.split()
    
    if len(original_words) < min_words or len(translation_words) < min_words:
        return False
        
    # Length ratio check
    len_ratio = min(len(original_words), len(translation_words)) / max(len(original_words), len(translation_words))
    if len_ratio < 0.5:  # Length difference too big
        return False
        
    # Calculate similarity
    similarity = calculate_similarity(original, translation)
    return similarity >= min_similarity

def extract_poetic_pairs(text: str) -> List[ParallelPair]:
    """Extract parallel pairs from poetic text."""
    lines = text.split('\n')
    pairs = []
    
    for i in range(0, len(lines)-1, 2):
        original = lines[i].strip()
        if i+1 < len(lines):
            translation = lines[i+1].strip()
            if is_valid_pair(original, translation):
                pairs.append(ParallelPair(
                    original=original,
                    translation=translation,
                    similarity=calculate_similarity(original, translation),
                    pair_type='poetic'
                ))
    
    return pairs

def extract_parallel_texts(text: str, min_similarity: float = 0.3) -> List[ParallelPair]:
    """Extract original and modern Turkish pairs from text."""
    paragraphs = text.split('\n\n')
    parallel_pairs = []

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        try:
            # Case 1: Direct parallel pairs (original followed by italicized translation)
            if i < len(paragraphs) - 1:
                next_para = paragraphs[i+1].strip()
                is_translation = (next_para.startswith('_') and 
                                next_para.endswith('_'))
                
                if is_translation:
                    original = para
                    translation = next_para.strip('_').strip()
                    
                    if is_valid_pair(original, translation, min_similarity=min_similarity):
                        parallel_pairs.append(ParallelPair(
                            original=original,
                            translation=translation,
                            similarity=calculate_similarity(original, translation),
                            pair_type='direct_pair'
                        ))

            # Case 2: Poetic stanzas
            elif '\n' in para:
                stanza_pairs = extract_poetic_pairs(para)
                parallel_pairs.extend(stanza_pairs)

        except Exception as e:
            logger.error(f"Error processing paragraph: {e}")
            continue

    return parallel_pairs

def verify_parallel_pairs(pairs: List[ParallelPair], 
                         similarity_threshold: float = 0.3) -> List[ParallelPair]:
    """Verify parallel pairs using similarity threshold."""
    return [
        pair for pair in pairs 
        if pair.similarity >= similarity_threshold
    ]

def save_parallel_pairs(pairs: List[ParallelPair], output_path: Path):
    """Save parallel pairs to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(f"Original: {pair.original}\n")
            f.write(f"Translation: {pair.translation}\n")
            f.write(f"Similarity: {pair.similarity:.3f}\n")
            f.write(f"Type: {pair.pair_type}\n")
            f.write("-" * 80 + "\n") 