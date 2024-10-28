import sys
import pathlib
import pymupdf4llm
from llama_index.core import SimpleDirectoryReader
import unicodedata
import re
import yaml
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
import traceback
from nltk.tokenize import sent_tokenize 
import json
import re
import string
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import matplotlib.pyplot as plt
from visualize import CorpusVisualizer
from network_visualizer import NetworkVisualizer  # Instead of CorpusVisualizer
try:
    import seaborn as sns
    plt.style.use('seaborn')
except (ImportError, OSError):
    # Fallback to a default style if seaborn isn't available
    plt.style.use('default')

def initialize_nltk():
    """Initialize NLTK by downloading required resources."""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 
                'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        nltk.download(resource, quiet=True)


def load_data_files():
    """Load data files with error handling."""
    try:
        current_dir = Path(__file__).parent
        data_dir = current_dir / 'data'
        
        # Initialize data dictionary
        data = {
            'OTTOMAN_STOPWORDS': set(),
            'SEMANTIC_FIELDS': {},
            'OTTOMAN_SUFFIXES': {},
            'REGISTER_MARKERS': {},
            'STYLE_MARKERS': {}
        }
        
        required_files = {
            'OTTOMAN_STOPWORDS': 'stopwords.json',
            'SEMANTIC_FIELDS': 'semantic_fields.json',
            'OTTOMAN_SUFFIXES': 'suffixes.json',
            'REGISTER_MARKERS': 'register_markers.json',
            'STYLE_MARKERS': 'style_markers.json'
        }
        
        for key, filename in required_files.items():
            try:
                file_path = data_dir / filename
                print(f"Loading {filename} from {file_path}")  # Debug print
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if key == 'OTTOMAN_STOPWORDS':
                        data[key] = set(loaded_data['particles_and_conjunctions'])
                    else:
                        data[key] = loaded_data
                print(f"Successfully loaded {key}")  # Debug print
                
            except FileNotFoundError:
                print(f"Warning: {filename} not found in {data_dir}")
            except json.JSONDecodeError:
                print(f"Warning: {filename} is not valid JSON")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        return data
    except Exception as e:
        print(f"Error in load_data_files: {str(e)}")
        return {
            'OTTOMAN_STOPWORDS': set(),
            'SEMANTIC_FIELDS': {},
            'OTTOMAN_SUFFIXES': {},
            'REGISTER_MARKERS': {},
            'STYLE_MARKERS': {}
        }

# Load data at module level
data = load_data_files()
OTTOMAN_STOPWORDS = data['OTTOMAN_STOPWORDS']
SEMANTIC_FIELDS = data['SEMANTIC_FIELDS']
OTTOMAN_SUFFIXES = data['OTTOMAN_SUFFIXES']
REGISTER_MARKERS = data['REGISTER_MARKERS']
STYLE_MARKERS = data['STYLE_MARKERS']

# Validate loaded data
print("\nValidating loaded data:")
print(f"OTTOMAN_SUFFIXES loaded: {bool(OTTOMAN_SUFFIXES)}")
print(f"REGISTER_MARKERS loaded: {bool(REGISTER_MARKERS)}")
print(f"Number of suffixes categories: {len(OTTOMAN_SUFFIXES)}")
print(f"Number of register markers: {len(REGISTER_MARKERS)}")

stats = {
    'document_counts': {
        'total': 0,
        'by_region': defaultdict(int),
        'by_period': defaultdict(int)
    },
    'token_statistics': {
        'total': 0,
        'by_region': defaultdict(int),
        'no_stopwords': {
            'total': 0,
            'by_region': defaultdict(int)
        }
    },
    'lexical_diversity': {
        'overall': 0.0,
        'by_region': defaultdict(list),
        'by_period': defaultdict(list)
    },
    'parallel_texts': {
        'total_pairs': 0,
        'by_region': defaultdict(list),
        'samples': []
    }
}


 ######################## ------------------------ ########################
 # Common Ottoman Turkish Stopwords                                       # 
 #       They are for experimental use only                               #
 #                                     update as needed                   #
 ######################## ------------------------ ########################

class SemanticAnalyzer:
    """Handles semantic analysis of Ottoman Turkish texts."""
    
    def __init__(self):
        self.semantic_fields = SEMANTIC_FIELDS  # Use the loaded JSON data
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

    def analyze_semantic_content(self, text: str) -> dict:
        """Analyze semantic field distribution in text."""
        words = word_tokenize(text.lower())
        field_scores = {}
        
        for field, terms in self.semantic_fields.items():
            field_words = set(words) & set(terms)
            field_scores[field] = len(field_words) / len(words) if words else 0
            
        return {
            'field_distribution': field_scores,
            'dominant_fields': sorted(
                field_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

    def calculate_semantic_preservation(self, original: str, translation: str) -> float:
        """Calculate semantic preservation score between original and translation."""
        try:
            # Create TF-IDF vectors
            vectors = self.vectorizer.fit_transform([original, translation])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0

    def identify_semantic_shifts(self, original: str, translation: str) -> Dict:
        """Identify semantic changes between versions."""
        original_fields = self.analyze_semantic_content(original)
        translation_fields = self.analyze_semantic_content(translation)
        
        shifts = {
            'field_changes': {},
            'preservation_score': self.calculate_semantic_preservation(original, translation),
            'semantic_distribution': {
                'original': original_fields,
                'translation': translation_fields
            }
        }
        
        # Calculate field-specific changes
        for field in self.semantic_fields:
            orig_score = original_fields.get(field, 0)
            trans_score = translation_fields.get(field, 0)
            shifts['field_changes'][field] = trans_score - orig_score
            
        return shifts
    
def calculate_lexical_diversity(text: str) -> float:
    """Calculate lexical diversity of text."""
    if not text:
        return 0.0
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def enhance_text_stats_with_semantics(text_stats: Dict, text: str) -> Dict:
    """Enhance text statistics with semantic analysis."""
    analyzer = SemanticAnalyzer()
    semantic_stats = analyzer.analyze_semantic_content(text)
    
    text_stats.update({
        'semantic_analysis': {
            'field_distribution': semantic_stats['field_distribution'],
            'dominant_fields': semantic_stats['dominant_fields']  # This is already sorted in analyze_semantic_content
        }
    })
    
    return text_stats

def determine_period(year):
    if year < 1600:
        return "Early Ottoman"
    elif 1600 <= year <= 1800:
        return "Classical Ottoman"
    else:
        return "Late Ottoman"


def create_metadata(pdf_path: pathlib.Path, region: str, text_stats: Dict) -> Dict:
    """Create enhanced metadata including semantic analysis."""
    year_match = re.search(r'_(\d{4})\.pdf$', str(pdf_path))
    year = int(year_match.group(1)) if year_match else None
    
    metadata = {
        'filename': pdf_path.name,
        'date': year,
        'period': determine_period(year) if year else 'Unknown',
        'region': region,
        'author': '',
        'title': '',
        'language': 'Ottoman Turkish',
        'genre': '',
        'statistics': {
            'token_count': text_stats['token_count'],
            'token_count_no_stopwords': text_stats['token_count_no_stopwords'],
            'unique_tokens': text_stats['unique_tokens'],
            'unique_tokens_no_stopwords': text_stats['unique_tokens_no_stopwords'],
            'lexical_diversity': text_stats['lexical_diversity'],
            'lexical_diversity_no_stopwords': text_stats['lexical_diversity_no_stopwords'],
            'stopwords_ratio': text_stats['stopwords_ratio'],
            'semantic_analysis': text_stats['semantic_analysis']
        }
    }
    return metadata

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using TF-IDF and cosine similarity.
    
    Args:
        text1: First text for comparison
        text2: Second text for comparison
        
    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        # Convert OTTOMAN_STOPWORDS set to list
        stopwords_list = list(OTTOMAN_STOPWORDS)
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=5000,    # Limit features to most frequent terms
            stop_words=stopwords_list  # Pass as list instead of set
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def calculate_levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Calculate normalized Levenshtein distance as a similarity measure.
    
    Args:
        text1: First text for comparison
        text2: Second text for comparison
        
    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        from Levenshtein import distance
        
        # Normalize texts
        text1 = ' '.join(word_tokenize(text1.lower()))
        text2 = ' '.join(word_tokenize(text2.lower()))
        
        # Calculate Levenshtein distance
        lev_distance = distance(text1, text2)
        
        # Normalize by the length of the longer string
        max_length = max(len(text1), len(text2))
        if max_length == 0:
            return 0.0
            
        similarity = 1 - (lev_distance / max_length)
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating Levenshtein similarity: {e}")
        return 0.0

def analyze_parallel_text_pair(original: str, translation: str) -> dict:
    """Analyze linguistic features of parallel text pairs with enhanced similarity metrics."""
    # Normalize both texts
    normalized_original = original
    normalized_translation = translation
    
    # Calculate basic token statistics
    orig_tokens = word_tokenize(normalized_original)
    trans_tokens = word_tokenize(normalized_translation)
    
    # Calculate multiple similarity metrics
    cosine_sim = calculate_similarity(normalized_original, normalized_translation)
    levenshtein_sim = calculate_levenshtein_similarity(normalized_original, normalized_translation)
    
    return {
        'original': {
            'text': original,
            'token_count': len(orig_tokens),
            'unique_tokens': len(set(orig_tokens)),
            'lexical_diversity': len(set(orig_tokens)) / len(orig_tokens) if orig_tokens else 0,
        },
        'translation': {
            'text': translation,
            'token_count': len(trans_tokens),
            'unique_tokens': len(set(trans_tokens)),
            'lexical_diversity': len(set(trans_tokens)) / len(trans_tokens) if trans_tokens else 0,
        },
        'pair_metrics': {
            'length_ratio': len(trans_tokens) / len(orig_tokens) if orig_tokens else 0,
            'cosine_similarity': cosine_sim,
            'levenshtein_similarity': levenshtein_sim,
            'shared_vocabulary': len(set(orig_tokens) & set(trans_tokens)),
            'shared_vocabulary_ratio': len(set(orig_tokens) & set(trans_tokens)) / 
                                     len(set(orig_tokens) | set(trans_tokens)) if orig_tokens and trans_tokens else 0
        }
    }

def save_parallel_texts(parallel_pairs, output_folder):
    """Save parallel texts in multiple formats."""
    if not parallel_pairs:
        return
        
    parallel_folder = output_folder / "parallel_texts"
    parallel_folder.mkdir(parents=True, exist_ok=True)
    
    # Save original texts
    with open(parallel_folder / "original.txt", 'w', encoding='utf-8') as f:
        for pair in parallel_pairs:
            f.write(f"{pair['original'].strip()}\n\n")
    
    # Save modern translations
    with open(parallel_folder / "modern.txt", 'w', encoding='utf-8') as f:
        for pair in parallel_pairs:
            f.write(f"{pair['translation'].strip()}\n\n")
    
    # Save summary statistics
    summary = {
        'total_pairs': len(parallel_pairs),
        'average_similarity': float(np.mean([p['similarity_score'] for p in parallel_pairs])),
        'pairs': [
            {
                'original': p['original'],
                'translation': p['translation'],
                'similarity': float(p['similarity_score'])
            }
            for p in parallel_pairs
        ]
    }
    
    with open(parallel_folder / "parallel_texts.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, allow_unicode=True, sort_keys=False)

def match_sentences(orig_sents, trans_sents):
    """Match sentences based on similarity scores."""
    matched_pairs = []
    
    for orig in orig_sents:
        best_match = None
        best_score = -1
        
        for trans in trans_sents:
            similarity = calculate_similarity(orig, trans)
            if similarity > best_score:
                best_score = similarity
                best_match = trans
        
        if best_match:
            matched_pairs.append((orig, best_match, best_score))
    
    return matched_pairs

def sent_tokenize(text):
    """Simple sentence tokenizer for Ottoman Turkish."""
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Clean up and filter empty sentences
    return [sent.strip() for sent in sentences if sent.strip()]
def create_enhanced_corpus_stats(stats: dict, base_output_path: pathlib.Path) -> None:
    """Create enhanced corpus statistics including morphological analysis."""
    print("\nCreating enhanced corpus statistics...")
    
    print("Processing document counts...")
    document_counts = {
        'by_period': dict(stats['document_counts']['by_period']),
        'by_region': dict(stats['document_counts']['by_region'])
    }
    
    print("Processing token statistics...")
    token_stats = {
        'by_period': dict(stats['token_statistics'].get('by_period', {})),
        'by_region': dict(stats['token_statistics']['by_region']),
        'no_stopwords': {
            'by_period': dict(stats['token_statistics']['no_stopwords'].get('by_period', {})),
            'by_region': dict(stats['token_statistics']['no_stopwords']['by_region'])
        }
    }
    
    print("Processing lexical diversity...")
    lex_diversity = {
        'by_region': dict(stats['lexical_diversity']['by_region']),
        'by_period': dict(stats['lexical_diversity']['by_period'])
    }
    
    corpus_stats = {
        'document_counts': document_counts,
        'token_statistics': token_stats,
        'lexical_diversity': lex_diversity,
        'parallel_texts': dict(stats['parallel_texts'])
    }
    
    print("Saving enhanced corpus stats...")
    stats_path = base_output_path / "md_out" / "enhanced_corpus_stats.yaml"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(corpus_stats, f, allow_unicode=True)
    print(f"Stats saved to: {stats_path}")
###############################################################
#        Parallel texts added for additional data analysis    #
###############################################################

def extract_parallel_texts(text) -> list:
    """Extract parallel texts based on formatting patterns."""
    # Handle Document objects or lists of Documents
    if hasattr(text, 'get_text'):
        text = text.get_text()
    elif isinstance(text, list):
        # If list contains Document objects, extract text from each
        text = ' '.join(doc.get_text() if hasattr(doc, 'get_text') else str(doc) 
                       for doc in text)
    elif not isinstance(text, str):
        text = str(text)
    
    pairs = []
    patterns = [
        r'Original:\s*(.*?)\s*Translation:\s*(.*?)(?=Original:|$)',
        r'Ottoman:\s*(.*?)\s*Modern:\s*(.*?)(?=Ottoman:|$)',
        r'Source:\s*(.*?)\s*Target:\s*(.*?)(?=Source:|$)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            original = match.group(1).strip()
            translation = match.group(2).strip()
            if original and translation:
                pairs.append((original, translation))
    
    return pairs

def is_valid_translation_pair(original: str, translation: str) -> bool:
    """
    Validate if two texts form a legitimate translation pair.
    
    Criteria:
    1. Similar length ratio
    2. Minimum similarity score
    3. Similar line count
    4. Similar punctuation pattern
    """
    if not original or not translation:
        return False
        
    # Length checks
    orig_tokens = word_tokenize(original)
    trans_tokens = word_tokenize(translation)
    if not orig_tokens or not trans_tokens:
        return False
        
    length_ratio = len(trans_tokens) / len(orig_tokens)
    if not (0.5 <= length_ratio <= 2.0):
        return False
        
    # Line count similarity
    orig_lines = original.count('\n') + 1
    trans_lines = translation.count('\n') + 1
    if abs(orig_lines - trans_lines) > 1:
        return False
        
    # Similarity check
    similarity = calculate_similarity(original, translation)
    if similarity < 0.3:
        return False
        
    # Punctuation pattern check
    orig_punct = ''.join(c for c in original if c in '.,!?;')
    trans_punct = ''.join(c for c in translation if c in '.,!?;')
    punct_ratio = len(trans_punct) / len(orig_punct) if orig_punct else 0
    if not (0.5 <= punct_ratio <= 2.0):
        return False
        
    return True

def is_valid_pair(original, translation):
    """Validate if the pair is legitimate."""
    if not original or not translation:
        return False
    if len(original.split()) < 3 or len(translation.split()) < 3:
        return False
    return True


def extract_poetic_pairs(stanza):
    """Extract parallel pairs from poetic stanzas."""
    lines = stanza.split('\n')
    pairs = []
    
    for i in range(0, len(lines)-1, 2):
        if i+1 < len(lines):
            original = lines[i].strip()
            translation = lines[i+1].strip('_').strip()
            if is_valid_pair(original, translation):
                pairs.append({
                    'original': original,
                    'translation': translation,
                    'type': 'poetic_pair'
                })
    return pairs

def clean_markdown_chunks(md_text):
    """Clean and process markdown text chunks."""
    cleaned_chunks = []
    for chunk in md_text:
        if 'text' in chunk:
            text = chunk['text']
            text = re.sub(r'\([0-9]+\)', '', text)
            text = re.sub(r'^\s*[0-9a-zA-Z]\.\s.*$', '', text, flags=re.MULTILINE)

            if chunk.get('type') == 'header':
                level = chunk.get('level', 1)
                text = f"{'#' * level} {text}"
            elif chunk.get('type') == 'list':
                text = f"- {text}"
            cleaned_chunks.append(text)
    return '\n\n'.join(cleaned_chunks)

def update_statistics(stats, metadata, region, text_stats):
    """Update corpus statistics."""
    try:
        # Update period counts
        period = metadata.get('period', 'Unknown')
        if period not in stats['document_counts']['by_period']:
            stats['document_counts']['by_period'][period] = 0
        stats['document_counts']['by_period'][period] += 1
        
        # Update token statistics
        stats['token_statistics']['total'] += text_stats['token_count']
        stats['token_statistics']['by_region'][region] = \
            stats['token_statistics']['by_region'].get(region, 0) + text_stats['token_count']
        
        # Update no-stopwords statistics
        stats['token_statistics']['no_stopwords']['total'] += text_stats['token_count_no_stopwords']
        stats['token_statistics']['no_stopwords']['by_region'][region] = \
            stats['token_statistics']['no_stopwords']['by_region'].get(region, 0) + text_stats['token_count_no_stopwords']
        
        # Update lexical diversity
        if 'lexical_diversity' in text_stats:
            stats['lexical_diversity']['by_region'][region] = \
                (stats['lexical_diversity']['by_region'].get(region, 0) + text_stats['lexical_diversity']) / 2
        
    except Exception as e:
        print(f"Error updating statistics: {str(e)}")
        traceback.print_exc()

def verify_parallel_pairs(pairs, similarity_threshold=0.3):
    """Verify parallel pairs using similarity threshold."""
    print(f"\nVerifying parallel pairs with threshold: {similarity_threshold}")
    print(f"Total pairs to verify: {len(pairs)}")
    verified_pairs = []
    
    for i, pair in enumerate(pairs):
        print(f"\nProcessing pair {i+1}/{len(pairs)}")
        original, translation = pair['original'], pair['translation']
        
        print("Normalizing texts...")
        normalized_original = normalize_text(original)
        normalized_translation = normalize_text(translation)
        
        print("Calculating similarity...")
        similarity = calculate_similarity(normalized_original, normalized_translation)
        print(f"Similarity score: {similarity:.3f}")
        
        if similarity >= similarity_threshold:
            print("Pair passed similarity threshold")
            orig_tokens = word_tokenize(original)
            trans_tokens = word_tokenize(translation)
            length_ratio = len(trans_tokens) / len(orig_tokens) if orig_tokens else 0
            print(f"Length ratio: {length_ratio:.3f}")
            
            verified_pairs.append({
                'original': original,
                'translation': translation,
                'similarity_score': similarity,
                'pair_metrics': {
                    'length_ratio': length_ratio,
                    'similarity': similarity
                }
            })
        else:
            print("Pair failed similarity threshold")

    print(f"\nVerification complete. Verified {len(verified_pairs)}/{len(pairs)} pairs")
    return verified_pairs

def calculate_morphological_complexity(texts):
    """Calculate morphological complexity of Ottoman Turkish texts."""
    try:
        # Use the already loaded OTTOMAN_SUFFIXES from data/suffixes.json
        if not isinstance(OTTOMAN_SUFFIXES, dict):
            print("Error: OTTOMAN_SUFFIXES not properly loaded")
            return 0.0
            
        print("\nCalculating morphological complexity...")
        print(f"Processing {len(texts)} texts")
        complexity_scores = []
        
        for i, text in enumerate(texts):
            print(f"\nProcessing text {i+1}/{len(texts)}")
            normalized_text = re.sub(r'[^\w\s]', '', text.lower())
            words = word_tokenize(normalized_text)
            
            if not words:
                print("Skipping empty text")
                continue
                
            morpheme_counts = []
            for word in words:
                count = 0
                # Iterate through all suffix categories in the loaded JSON
                for category, suffix_list in OTTOMAN_SUFFIXES.items():
                    for suffix in suffix_list:
                        if word.endswith(suffix):
                            count += 1
                morpheme_counts.append(count)
            
            if morpheme_counts:
                avg_complexity = np.mean(morpheme_counts)
                complexity_scores.append(avg_complexity)
        
        final_score = np.mean(complexity_scores) if complexity_scores else 0.0
        return final_score
        
    except Exception as e:
        print(f"Error in morphological complexity calculation: {str(e)}")
        return 0.0

def analyze_text_complexity(text: str) -> dict:
    """Analyze text complexity including morphological features."""
    print("\nAnalyzing text complexity...")
    print(f"Input text length: {len(text)}")
    
    morph_complexity = calculate_morphological_complexity([text])
    print(f"Morphological complexity: {morph_complexity:.3f}")
    
    words = word_tokenize(text)
    avg_word_length = np.mean([len(word) for word in words])
    print(f"Average word length: {avg_word_length:.3f}")
    
    print("Analyzing suffix distribution...")
    suffix_dist = analyze_suffix_distribution(text)
    print(f"Suffix distribution: {suffix_dist}")
    
    return {
        'morphological_complexity': morph_complexity,
        'average_word_length': avg_word_length,
        'suffix_distribution': suffix_dist
    }

def analyze_suffix_distribution(text: str) -> dict:
    """Analyze distribution of different suffix types."""
    print("\nAnalyzing suffix distribution...")
    normalized_text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(normalized_text)
    print(f"Processing {len(words)} words")
    
    suffix_counts = defaultdict(int)
    for i, word in enumerate(words):
        if i % 1000 == 0:  # Progress update
            print(f"Processing word {i+1}/{len(words)}")
        for category, suffixes in OTTOMAN_SUFFIXES.items():
            for suffix in suffixes:
                if word.endswith(suffix):
                    suffix_counts[category] += 1
                    
    print("Suffix analysis complete")
    print(f"Found suffixes in categories: {dict(suffix_counts)}")
    return dict(suffix_counts)

def create_enhanced_corpus_stats(stats: dict, base_output_path: pathlib.Path) -> None:
    """Create enhanced corpus statistics including morphological analysis."""
    print("\nCreating enhanced corpus statistics...")
    
    print("Processing document counts...")
    document_counts = {
        'by_period': dict(stats['document_counts']['by_period']),
        'by_region': dict(stats['document_counts']['by_region'])
    }
    
    print("Processing token statistics...")
    token_stats = {
        'by_period': dict(stats['token_statistics'].get('by_period', {})),
        'by_region': dict(stats['token_statistics']['by_region']),
        'no_stopwords': {
            'by_period': dict(stats['token_statistics']['no_stopwords'].get('by_period', {})),
            'by_region': dict(stats['token_statistics']['no_stopwords']['by_region'])
        }
    }
    
    print("Processing lexical diversity...")
    lex_diversity = {
        'by_region': dict(stats['lexical_diversity']['by_region']),
        'by_period': dict(stats['lexical_diversity']['by_period'])
    }
    
    corpus_stats = {
        'document_counts': document_counts,
        'token_statistics': token_stats,
        'lexical_diversity': lex_diversity,
        'parallel_texts': dict(stats['parallel_texts'])
    }
    
    print("Saving enhanced corpus stats...")
    stats_path = base_output_path / "md_out" / "enhanced_corpus_stats.yaml"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(corpus_stats, f, allow_unicode=True)
    print(f"Stats saved to: {stats_path}")

def plot_morphological_complexity(self):
    """Create visualization for morphological complexity across regions."""
    print("\nPlotting morphological complexity...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("Setting up plot...")
    plt.figure(figsize=(12, 6))
    
    regions = list(self.stats['morphological_analysis']['by_region'].keys())
    complexity_scores = [self.stats['morphological_analysis']['by_region'][r] for r in regions]
    print(f"Plotting data for {len(regions)} regions")
    
    print("Creating bar plot...")
    sns.barplot(x=regions, y=complexity_scores)
    plt.title('Morphological Complexity by Region')
    plt.xlabel('Region')
    plt.ylabel('Complexity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print("Saving plot...")
    plt.savefig('morphological_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as morphological_complexity.png")

def pdf_to_markdown(pdf_file: pathlib.Path, stats: dict, semantic: bool = False) -> bool:
    """Process PDF file and extract parallel texts."""
    print(f"\nProcessing PDF file: {pdf_file}")
    
    try:
        print("Extracting text from PDF...")
        region = pdf_file.parent.name
        
        # Extract period from filename (e.g., "Text_1800.pdf" -> "1800")
        period = None
        filename = pdf_file.stem
        year_match = re.search(r'_(\d{4})', filename)
        if year_match:
            period = year_match.group(1)
        
        # Initialize text stats with all required fields
        text_stats = {
            'token_count': 0,
            'token_count_no_stopwords': 0,
            'unique_tokens_no_stopwords': 0,
            'lexical_diversity_no_stopwords': 0.0,
            'stopwords_ratio': 0.0,
            'semantic_analysis': {},
            'unique_tokens': 0,
            'sentence_count': 0,
            'lexical_diversity': 0.0,
            'semantic_fields': {},
            'style_markers': {},
            'period': period,
            'morphological_complexity': 0.0,
            'average_word_length': 0.0,
            'punctuation_density': 0.0,
            'formality_score': 0.0
        }
        
        # Extract text using PyMuPDF
        import fitz
        try:
            doc = fitz.open(str(pdf_file))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return False
        
        # Calculate basic stats
        tokens = word_tokenize(text)
        unique_tokens = set(tokens)
        sentences = sent_tokenize(text)
        
        # Calculate stats without stopwords
        tokens_no_stopwords = [
            token.lower() for token in tokens 
            if token.lower() not in OTTOMAN_STOPWORDS
        ]
        unique_tokens_no_stopwords = set(tokens_no_stopwords)
        
        # Calculate additional metrics
        if tokens:
            stopwords_ratio = (len(tokens) - len(tokens_no_stopwords)) / len(tokens)
            average_word_length = np.mean([len(token) for token in tokens])
            lexical_diversity = len(unique_tokens) / len(tokens)
            lexical_diversity_no_stopwords = len(unique_tokens_no_stopwords) / len(tokens_no_stopwords) if tokens_no_stopwords else 0.0
        else:
            stopwords_ratio = 0.0
            average_word_length = 0.0
            lexical_diversity = 0.0
            lexical_diversity_no_stopwords = 0.0
        
        # Calculate morphological complexity
        morphological_complexity = calculate_morphological_complexity([text])
        
        # Calculate punctuation density
        punct_density = calculate_punctuation_density([text])
        
        # Calculate formality score
        formality_score = calculate_register_score(tokens)
        
        # Update text stats
        text_stats.update({
            'token_count': len(tokens),
            'token_count_no_stopwords': len(tokens_no_stopwords),
            'unique_tokens': len(unique_tokens),
            'unique_tokens_no_stopwords': len(unique_tokens_no_stopwords),
            'sentence_count': len(sentences),
            'lexical_diversity': lexical_diversity,
            'lexical_diversity_no_stopwords': lexical_diversity_no_stopwords,
            'stopwords_ratio': stopwords_ratio,
            'average_word_length': average_word_length,
            'morphological_complexity': morphological_complexity,
            'punctuation_density': punct_density['average_density'],
            'formality_score': formality_score
        })
        
        # Create metadata with the required arguments
        metadata = create_metadata(pdf_file, region, text_stats)
        
        # Update global stats
        # Initialize region entries if they don't exist
        for stat_category in ['document_counts', 'token_statistics', 'lexical_diversity']:
            if region not in stats[stat_category]['by_region']:
                stats[stat_category]['by_region'][region] = 0 if stat_category != 'lexical_diversity' else []
        
        if region not in stats['token_statistics']['no_stopwords']['by_region']:
            stats['token_statistics']['no_stopwords']['by_region'][region] = 0
            
        # Update counts
        stats['document_counts']['total'] += 1
        stats['document_counts']['by_region'][region] += 1
        stats['token_statistics']['total'] += text_stats['token_count']
        stats['token_statistics']['by_region'][region] += text_stats['token_count']
        stats['token_statistics']['no_stopwords']['total'] += text_stats['token_count_no_stopwords']
        stats['token_statistics']['no_stopwords']['by_region'][region] += text_stats['token_count_no_stopwords']
        
        # Update lexical diversity
        if text_stats['lexical_diversity'] > 0:
            stats['lexical_diversity']['by_region'][region].append(text_stats['lexical_diversity'])
        
        # Update temporal statistics if period exists
        if period:
            # Initialize period entries if they don't exist
            if 'by_period' not in stats['document_counts']:
                stats['document_counts']['by_period'] = {}
            if period not in stats['document_counts']['by_period']:
                stats['document_counts']['by_period'][period] = 0
            
            # Update period counts
            stats['document_counts']['by_period'][period] += 1
            
            # Update regional-temporal matrix
            if 'regional_temporal_matrix' not in stats:
                stats['regional_temporal_matrix'] = {}
            if region not in stats['regional_temporal_matrix']:
                stats['regional_temporal_matrix'][region] = {}
            if period not in stats['regional_temporal_matrix'][region]:
                stats['regional_temporal_matrix'][region][period] = 0
            stats['regional_temporal_matrix'][region][period] += 1
            
            # Update lexical diversity by period
            if 'by_period' not in stats['lexical_diversity']:
                stats['lexical_diversity']['by_period'] = {}
            if period not in stats['lexical_diversity']['by_period']:
                stats['lexical_diversity']['by_period'][period] = []
            stats['lexical_diversity']['by_period'][period].append(text_stats['lexical_diversity'])
        
        # Update linguistic analysis stats
        if 'linguistic_analysis' not in stats:
            stats['linguistic_analysis'] = {
                'morphological_complexity': {'by_region': {}, 'by_period': {}},
                'formality_scores': {'by_region': {}, 'by_period': {}},
                'average_word_length': {'by_region': {}, 'by_period': {}}
            }
        
        # Update regional linguistic stats
        for metric in ['morphological_complexity', 'formality_scores', 'average_word_length']:
            if region not in stats['linguistic_analysis'][metric]['by_region']:
                stats['linguistic_analysis'][metric]['by_region'][region] = []
            stats['linguistic_analysis'][metric]['by_region'][region].append(
                text_stats[metric.replace('_scores', '_score')]
            )
        
        # Update temporal linguistic stats if period exists
        if period:
            for metric in ['morphological_complexity', 'formality_scores', 'average_word_length']:
                if period not in stats['linguistic_analysis'][metric]['by_period']:
                    stats['linguistic_analysis'][metric]['by_period'][period] = []
                stats['linguistic_analysis'][metric]['by_period'][period].append(
                    text_stats[metric.replace('_scores', '_score')]
                )
        
        print(f"Successfully processed {pdf_file.name}")
        try:
            text = pymupdf4llm.LlamaMarkdownReader().load_data(str(pdf_file))
            parallel_pairs = extract_parallel_texts(text)
            
            if parallel_pairs:
                print(f"Found {len(parallel_pairs)} parallel text pairs")
                region = pdf_file.parent.name
                
                # Ensure parallel_texts structure exists with dynamic categories
                if 'parallel_texts' not in stats:
                    stats['parallel_texts'] = {
                        'total_pairs': 0,
                        'by_region': defaultdict(list),
                        'by_category': defaultdict(list),
                        'samples': [],
                        'metadata': defaultdict(dict)
                    }
                
                # Process each pair
                for original, translation in parallel_pairs:
                    # Analyze the pair
                    analysis = analyze_parallel_text_pair(original, translation)
                    
                    # Dynamically categorize based on analysis
                    categories = analysis.get('categories', ['uncategorized'])
                    
                    # Update stats with flexible structure
                    stats['parallel_texts']['total_pairs'] += 1
                    
                    # Store by region
                    stats['parallel_texts']['by_region'][region].append({
                        'original': original,
                        'translation': translation,
                        'analysis': analysis
                    })
                    
                    # Store by category
                    for category in categories:
                        stats['parallel_texts']['by_category'][category].append({
                            'original': original,
                            'translation': translation,
                            'analysis': analysis,
                            'region': region
                        })
                    
                    # Store representative samples
                    if len(stats['parallel_texts']['samples']) < 10:
                        stats['parallel_texts']['samples'].append({
                            'region': region,
                            'original': original,
                            'translation': translation,
                            'metrics': analysis['pair_metrics'],
                            'categories': categories
                        })
        
            return True
        
        except Exception as e:
            print(f"Error processing parallel texts: {str(e)}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        traceback.print_exc()
        return False
    
    
    
def calculate_register_score(tokens):
    """Calculate formality/register score based on linguistic markers."""
    try:
        print("\nCalculating register score...")
        
        # Use the already loaded REGISTER_MARKERS from data files
        if not isinstance(REGISTER_MARKERS, dict):
            print("Error: REGISTER_MARKERS not properly loaded")
            return 0.0
        
        # Get formal and informal markers
        formal_markers = set(REGISTER_MARKERS.get('formal_markers', []))
        informal_markers = set(REGISTER_MARKERS.get('informal_markers', []))
        
        if not tokens:
            return 0.0
            
        # Convert tokens to lowercase for comparison
        tokens_lower = [token.lower() for token in tokens]
        
        # Count markers
        formal_count = sum(1 for token in tokens_lower if token in formal_markers)
        informal_count = sum(1 for token in tokens_lower if token in informal_markers)
        
        # Calculate score (-1 to 1, where 1 is most formal)
        total_markers = formal_count + informal_count
        if total_markers == 0:
            return 0.0
            
        score = (formal_count - informal_count) / total_markers
        print(f"Register score: {score:.3f} (formal: {formal_count}, informal: {informal_count})")
        return score
        
    except Exception as e:
        print(f"Error calculating register score: {str(e)}")
        traceback.print_exc()  # This will help debug any issues
        return 0.0
    

def normalize_text(text):
    """Normalize text using character mappings."""
    print("\nNormalizing text...")
    print(f"Input text length: {len(text)}")
    char_setup = {}  # Add your character mappings here
    print(f"Using character mappings: {char_setup}")
    
    for old, new in char_setup.items():
        text = text.replace(old, new)
    
    print(f"Normalized text length: {len(text)}")
    return text

def is_poetry(text):
    """Detect if text is poetry based on common patterns."""
    print("\nChecking if text is poetry...")
    poetry_markers = [
        'Müfteilün', 'Fâilün', 'Mefâilün', 'Feilâtün',  # Common aruz meters
        '###',  # Header markers for verses
        '\n\n',  # Multiple line breaks between stanzas
    ]
    print(f"Checking for poetry markers: {poetry_markers}")
    
    for marker in poetry_markers:
        if marker in text:
            print(f"Found poetry marker: {marker}")
            return True
    
    print("No poetry markers found")
    return False

def calculate_punctuation_density(texts):
    """Calculate punctuation density of texts."""
    print("\nCalculating punctuation density...")
    punctuation_marks = set('.,!?;:""''()[]{}«»-–—')
    print(f"Using punctuation marks: {punctuation_marks}")
    
    densities = []
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")
        total_chars = len(text)
        print(f"Total characters: {total_chars}")
        
        if total_chars == 0:
            print("Skipping empty text")
            continue
            
        punct_count = sum(1 for char in text if char in punctuation_marks)
        density = punct_count / total_chars
        print(f"Punctuation count: {punct_count}, Density: {density:.4f}")
        densities.append(density)
    
    result = {
        'average_density': np.mean(densities) if densities else 0.0,
        'punctuation_distribution': Counter(char for text in texts 
                                         for char in text if char in punctuation_marks)
    }
    print(f"Final results: {result}")
    return result

def identify_formality_markers(words):
    """Identify markers of formality in text."""
    print("\nIdentifying formality markers...")
    formality_markers = []
    
    formal_patterns = {
        'buyurmak': 1.0,
        'teşrif': 1.0,
        'istirham': 1.0,
        'bendeniz': 1.0,
        'zat-ı': 1.0
    }
    print(f"Checking for patterns: {list(formal_patterns.keys())}")
    
    for i, word in enumerate(words):
        if i % 1000 == 0:  # Progress update every 1000 words
            print(f"Processing word {i+1}/{len(words)}")
        word_lower = word.lower()
        for pattern, score in formal_patterns.items():
            if pattern in word_lower:
                print(f"Found formal marker: {word} (pattern: {pattern}, score: {score})")
                formality_markers.append((word, score))
    
    print(f"Total formality markers found: {len(formality_markers)}")
    return formality_markers

def identify_style_markers(texts, text_type):
    """Identify style markers in texts."""
    print(f"\nIdentifying style markers for {text_type} texts...")
    markers_found = defaultdict(int)
    
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")
        words = word_tokenize(text.lower())
        print(f"Words tokenized: {len(words)}")
        
        for category, markers in STYLE_MARKERS.items():
            print(f"Checking category: {category}")
            if isinstance(markers, dict):
                for subcategory, submarkers in markers.items():
                    print(f"  Checking subcategory: {subcategory}")
                    for marker in submarkers:
                        count = sum(1 for word in words if marker in word.lower())
                        if count > 0:
                            print(f"    Found marker '{marker}': {count} times")
                        markers_found[f"{category}_{subcategory}"] += count
            else:
                for marker in markers:
                    count = sum(1 for word in words if marker in word.lower())
                    if count > 0:
                        print(f"  Found marker '{marker}': {count} times")
                    markers_found[category] += count
    
    result = dict(markers_found)
    print(f"Final style marker counts: {result}")
    return result


def analyze_text_features(texts, text_type):
    """Analyze linguistic features of texts."""
    print(f"\nAnalyzing text features for type: {text_type}")
    print(f"Number of texts to analyze: {len(texts)}")
    
    try:
        print("Tokenizing words...")
        words = [word for text in texts for word in word_tokenize(text.lower())]
        print(f"Total words tokenized: {len(words)}")
        
        print("Calculating word frequencies...")
        word_counts = Counter(words)
        print(f"Unique words found: {len(word_counts)}")
        
        print("Building vocabulary statistics...")
        vocabulary_stats = {
            'total_tokens': len(words),
            'unique_tokens': len(set(words)),
            'hapax_legomena': len([w for w, c in word_counts.items() if c == 1]),
            'average_word_length': np.mean([len(w) for w in words]),
            'most_frequent_words': dict(word_counts.most_common(20))
        }
        print(f"Vocabulary stats calculated: {vocabulary_stats}")
        
        print("Analyzing structural features...")
        structural_stats = {
            'average_sentence_length': np.mean([len(sent.split()) 
                for text in texts for sent in text.split('.')]),
            'morphological_complexity': calculate_morphological_complexity(texts),
            'punctuation_density': calculate_punctuation_density(texts)
        }
        print(f"Structural analysis complete: {structural_stats}")
        
        print("Identifying style markers...")
        style_markers = identify_style_markers(texts, text_type)
        print(f"Style markers identified: {style_markers}")
        
        return {
            'vocabulary': vocabulary_stats,
            'structural': structural_stats,
            'style_markers': style_markers
        }
    except Exception as e:
        print(f"Error in text feature analysis: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        return {
            'vocabulary': {'total_tokens': 0, 'unique_tokens': 0, 'hapax_legomena': 0,
                         'average_word_length': 0, 'most_frequent_words': {}},
            'structural': {'average_sentence_length': 0, 'morphological_complexity': 0,
                         'punctuation_density': 0},
            'style_markers': {}
        }

def analyze_parallel_texts(parallel_pairs):
    """Comprehensive analysis of parallel texts."""
    print(f"\nAnalyzing parallel texts...")
    print(f"Number of parallel pairs: {len(parallel_pairs)}")
    
    try:
        print("Calculating overall statistics...")
        similarity_scores = [p['similarity_score'] for p in parallel_pairs]
        print(f"Average similarity score: {np.mean(similarity_scores):.3f}")
        
        print("Identifying text types...")
        poetry_count = sum(1 for p in parallel_pairs if is_poetry(p['original']))
        prose_count = len(parallel_pairs) - poetry_count
        print(f"Poetry pairs: {poetry_count}, Prose pairs: {prose_count}")
        
        print("Analyzing original texts...")
        original_features = analyze_text_features([p['original'] for p in parallel_pairs], 'original')
        
        print("Analyzing modern translations...")
        modern_features = analyze_text_features([p['translation'] for p in parallel_pairs], 'modern')
        
        analysis = {
            'overall_statistics': {
                'pair_count': len(parallel_pairs),
                'average_similarity': np.mean(similarity_scores),
                'text_types': {
                    'poetry': poetry_count,
                    'prose': prose_count
                }
            },
            'linguistic_features': {
                'original': original_features,
                'modern': modern_features
            }
        }
        print("Analysis complete!")
        return analysis
        
    except Exception as e:
        print(f"Error in parallel text analysis: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        return {
            'overall_statistics': {
                'pair_count': 0,
                'average_similarity': 0,
                'text_types': {'poetry': 0, 'prose': 0}
            },
            'linguistic_features': {
                'original': {},
                'modern': {}
            }
        }
# Add other helper functions (analyze_text_features, clean_markdown_chunks, etc.)

def process_parallel_pair(original_pdf: pathlib.Path, translation_pdf: pathlib.Path, stats: dict) -> None:
    """Process a pair of parallel texts (original and translation)."""
    try:
        # Update these lines to use extract_text
        original_text = pymupdf4llm.extract_text(str(original_pdf))  # Changed from get_text to extract_text
        translation_text = pymupdf4llm.extract_text(str(translation_pdf))  # Changed from get_text to extract_text
        
        analysis = analyze_parallel_text_pair(original_text, translation_text)
        stats['parallel_texts'].append(analysis)
        stats['parallel_metrics']['similarity_scores'].append(analysis['pair_metrics']['cosine_similarity'])
        stats['parallel_metrics']['length_ratios'].append(analysis['pair_metrics']['length_ratio'])
        stats['parallel_metrics']['shared_vocabulary_ratios'].append(analysis['pair_metrics']['shared_vocabulary_ratio'])
        
        # Process original file stats
        pdf_to_markdown(original_pdf, stats)
        
    except Exception as e:
        print(f"Error processing parallel pair: {str(e)}")
        traceback.print_exc()


def process_directory(input_dir, parallel=False, semantic=False, morphological=False):
    """Process directory of PDFs."""
    try:
        # Initialize stats with regular dictionaries instead of defaultdict
        stats = {
            'document_counts': {
                'total': 0,
                'by_region': {},
                'by_period': {}
            },
            'token_statistics': {
                'total': 0,
                'by_region': {},
                'no_stopwords': {
                    'total': 0,
                    'by_region': {}
                }
            },
            'lexical_diversity': {
                'overall': 0.0,
                'by_region': {},
                'by_period': {}
            },
            'parallel_texts': {
                'total_pairs': 0,
                'by_region': defaultdict(list),
                'samples': []
            }
        }
        
        # Process files
        processed_count = 0
        processed_files = set()  # Keep track of processed files
        
        for pdf_file in pathlib.Path(input_dir).rglob('*.pdf'):
            if pdf_file in processed_files:  # Skip if already processed
                continue
                
            try:
                print(f"\nProcessing: {pdf_file}")
                result = pdf_to_markdown(pdf_file, stats, semantic=semantic)
                if result:
                    processed_files.add(pdf_file)  # Add to processed set
                    processed_count += 1
                    region = pdf_file.parent.name
                    
                    # Initialize region if not exists
                    if region not in stats['document_counts']['by_region']:
                        stats['document_counts']['by_region'][region] = 0
                        stats['token_statistics']['by_region'][region] = 0
                        stats['token_statistics']['no_stopwords']['by_region'][region] = 0
                        stats['lexical_diversity']['by_region'][region] = 0.0
                        stats['parallel_texts']['by_region'][region] = []
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        # Convert any remaining defaultdicts to regular dicts
        stats = convert_defaultdict_to_dict(stats)
        
        print("\nProcessing completed successfully!")
        print(f"Files processed: {processed_count}")
        print(f"\nParallel texts found: {stats['parallel_texts']['total_pairs']}")
        
        return stats
        
    except Exception as e:
        print(f"Error in process_directory: {str(e)}")
        traceback.print_exc()
        return None
    

def convert_defaultdict_to_dict(obj):
    """Convert defaultdict to regular dict recursively."""
    print(f"Converting object of type: {type(obj)}")
    if isinstance(obj, (defaultdict, dict)):
        print(f"Converting dictionary with keys: {list(obj.keys())}")
        return {k: convert_defaultdict_to_dict(v) for k, v in dict(obj).items()}
    elif isinstance(obj, list):
        print(f"Converting list of length: {len(obj)}")
        return [convert_defaultdict_to_dict(v) for v in obj]
    print(f"Returning primitive value: {obj}")
    return obj

def find_translation_pair(pdf_file):
    """Find translation pair for a given PDF file."""
    print(f"\nSearching for translation pair for: {pdf_file}")
    stem = pdf_file.stem
    print(f"File stem: {stem}")
    
    # Look for files with similar names in the same directory
    print(f"Searching in directory: {pdf_file.parent}")
    for potential_pair in pdf_file.parent.glob('*.pdf'):
        print(f"Checking potential pair: {potential_pair.name}")
        if potential_pair != pdf_file and (
            potential_pair.stem.startswith(stem) or 
            stem.startswith(potential_pair.stem) or
            'translation' in potential_pair.stem.lower() or
            'tercüme' in potential_pair.stem.lower()
        ):
            print(f"Found matching pair: {potential_pair}")
            return potential_pair
    print("No translation pair found")
    return None
def process_document_stats(text, region, stats):
    """Process and store comprehensive document statistics."""
    try:
        # Token statistics
        tokens = word_tokenize(text)
        tokens_no_stop = [t for t in tokens if t not in OTTOMAN_STOPWORDS]
        
        # Update token counts
        stats['token_statistics']['by_region'][region] = len(tokens)
        stats['token_statistics']['no_stopwords']['by_region'][region] = len(tokens_no_stop)
        
        # Morphological analysis
        if 'morphological_complexity' not in stats:
            stats['morphological_complexity'] = {'by_region': defaultdict(list)}
        morph_score = calculate_morphological_complexity(text)
        stats['morphological_complexity']['by_region'][region].append(morph_score)
        
        # Register and style markers
        if 'style_markers' not in stats:
            stats['style_markers'] = {'by_region': defaultdict(list)}
        register_score = calculate_register_score(tokens)
        style_markers = identify_style_markers(text)
        stats['style_markers']['by_region'][region].extend(style_markers)
        
        # Semantic fields
        if 'semantic_fields' not in stats:
            stats['semantic_fields'] = {'by_region': defaultdict(set)}
        semantic_fields = identify_semantic_fields(text)
        stats['semantic_fields']['by_region'][region].update(semantic_fields)
        
        print(f"\nDocument Statistics for {region}:")
        print(f"- Tokens: {len(tokens)}")
        print(f"- Tokens (no stopwords): {len(tokens_no_stop)}")
        print(f"- Morphological complexity: {morph_score:.3f}")
        print(f"- Register score: {register_score:.3f}")
        print(f"- Semantic fields: {len(semantic_fields)}")
        
    except Exception as e:
        print(f"Error processing document stats: {e}")
        traceback.print_exc()

def identify_style_markers(text):
    """Identify style markers in text using REGISTER_MARKERS."""
    markers = []
    words = word_tokenize(text.lower())
    
    for word in words:
        # Check formal markers
        if word in REGISTER_MARKERS.get('formal_markers', []):
            markers.append(('formal', word))
        
        # Check informal markers
        if word in REGISTER_MARKERS.get('informal_markers', []):
            markers.append(('informal', word))
            
        # Check greetings
        for formality in ['formal', 'informal']:
            if word in REGISTER_MARKERS.get('greetings', {}).get(formality, []):
                markers.append((f'greeting_{formality}', word))
                
        # Check farewells
        for formality in ['formal', 'informal']:
            if word in REGISTER_MARKERS.get('farewells', {}).get(formality, []):
                markers.append((f'farewell_{formality}', word))
    
    return markers

def identify_semantic_fields(text):
    """Identify semantic fields in text using SEMANTIC_FIELDS."""
    fields = set()
    words = word_tokenize(text.lower())
    
    for word in words:
        for field, terms in SEMANTIC_FIELDS.items():
            if word in terms:
                fields.add(field)
    
    return fields

def create_output_structure(pdf_file: pathlib.Path) -> pathlib.Path:
    """Create output directory structure for processed files."""
    print(f"\nCreating output structure for: {pdf_file}")
    try:
        # Create base output directory parallel to PDF directory
        base_dir = pdf_file.parent.parent / "md_out"
        
        # Create region-based subdirectory structure
        region_dir = base_dir / pdf_file.parent.name
        
        # Create specific output directory for this file
        output_dir = region_dir / pdf_file.stem
        
        # Create all directories if they don't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created output directory: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"Error creating output structure: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import argparse

    initialize_nltk()
    
    # Define output_dir before using it
    base_output_path = pathlib.Path(r"C:\Users\Administrator\Desktop\cook\Ottoman-NLP\corpus-texts")
    output_dir = base_output_path / "md_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating lexical network visualization...")
    network_viz = NetworkVisualizer(str(output_dir / "enhanced_corpus_stats.yaml"))
    network_viz.generate_network_analysis(output_dir=str(output_dir / "network_analysis"))
    
    description = """
    Thank you for using OttoMiner!
    
    This tool processes Ottoman Turkish PDF files and performs various analyses including:
    - Parallel text processing
    - Semantic analysis
    - Morphological analysis
    - Visualization generation

    Example usage:
    python pdf_to_md.py /path/to/pdfs -p -s -m -g
    """
    
    epilog = """
    For more information, visit: https://github.com/your-repo/ottominer
    """
    
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='OttoMiner'
    )
    
    # Required argument
    parser.add_argument(
        'input_directory',
        help='Directory containing Ottoman Turkish PDF files to process'
    )
    
    # Optional arguments
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Enable parallel text processing and alignment'
    )
    analysis_group.add_argument(
        '-s', '--semantic',
        action='store_true',
        help='Enable semantic field analysis and classification'
    )
    analysis_group.add_argument(
        '-m', '--morphological',
        action='store_true',
        help='Enable morphological analysis of Ottoman Turkish texts'
    )
    
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-g', '--graph',
        action='store_true',
        help='Generate visualization graphs of analysis results'
    )
    
    args = parser.parse_args()
    
    try:
        base_output_path = pathlib.Path(r"C:\Users\Administrator\Desktop\cook\Ottoman-NLP\corpus-texts")
        output_dir = base_output_path / "md_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = process_directory(
            args.input_directory,
            parallel=args.parallel,
            semantic=args.semantic,
            morphological=args.morphological
        )
        
        create_enhanced_corpus_stats(stats, base_output_path)
        
        print("\nOutput saved in:", output_dir)
        if args.morphological:
            print("\nMorphological Analysis Results:")
            for region, complexity in stats.get('morphological_analysis', {}).items():
                print(f"{region}: {np.mean(complexity):.3f}")
        
        if args.graph:
            from visualize import CorpusVisualizer
            visualizer = CorpusVisualizer(str(output_dir / "enhanced_corpus_stats.yaml"))
            plots_dir = output_dir / "visualization_plots"
            visualizer.generate_all_plots(output_dir=str(plots_dir))
            print(f"\nVisualization graphs generated in {plots_dir}")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        traceback.print_exc()

