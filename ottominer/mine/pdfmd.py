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


def initialize_nltk():
    """Initialize NLTK by downloading required resources."""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 
                'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        nltk.download(resource, quiet=True)


def load_data_files():
    current_dir = Path(__file__).parent
    
    # Load all data files
    with open(current_dir / 'data' / 'stopwords.json', 'r', encoding='utf-8') as f:
        OTTOMAN_STOPWORDS = set(json.load(f)['particles_and_conjunctions'])
    
    with open(current_dir / 'data' / 'semantic_fields.json', 'r', encoding='utf-8') as f:
        SEMANTIC_FIELDS = json.load(f)
    
    with open(current_dir / 'data' / 'suffixes.json', 'r', encoding='utf-8') as f:
        OTTOMAN_SUFFIXES = json.load(f)
    
    with open(current_dir / 'data' / 'register_markers.json', 'r', encoding='utf-8') as f:
        REGISTER_MARKERS = json.load(f)
        
    with open(current_dir / 'data' / 'style_markers.json', 'r', encoding='utf-8') as f:
        STYLE_MARKERS = json.load(f)
    
    return OTTOMAN_STOPWORDS, SEMANTIC_FIELDS, OTTOMAN_SUFFIXES, REGISTER_MARKERS, STYLE_MARKERS

# Load data at module level
OTTOMAN_STOPWORDS, SEMANTIC_FIELDS, OTTOMAN_SUFFIXES, REGISTER_MARKERS, STYLE_MARKERS = load_data_files()

# Load data at module level
OTTOMAN_STOPWORDS, SEMANTIC_FIELDS, OTTOMAN_SUFFIXES = load_data_files()

class SemanticAnalyzer:
    """Handles semantic analysis of Ottoman Turkish texts."""
    
    def __init__(self):
        self.semantic_fields = SEMANTIC_FIELDS
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

stats = {
    'period_counts': defaultdict(int),
    'region_counts': defaultdict(int),
    'token_counts': defaultdict(int),
    'tokens_by_region': defaultdict(int),
    'token_counts_no_stopwords': defaultdict(int),
    'tokens_by_region_no_stopwords': defaultdict(int),
    'lexical_diversity': defaultdict(list),
    'lexical_diversity_no_stopwords': defaultdict(list),
    'lexical_diversity_by_period': defaultdict(list),
    'lexical_diversity_no_stopwords_by_period': defaultdict(list),
    'stopwords_ratios': defaultdict(list),
    'parallel_texts': [],
    'parallel_metrics': {
        'similarity_scores': [],
        'length_ratios': [],
        'shared_vocabulary_ratios': []
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

def create_enhanced_corpus_stats(stats, base_output_path):
    """Create enhanced corpus statistics including parallel text analysis."""
    corpus_stats = {
        'document_counts': {
            'by_period': dict(stats['period_counts']),
            'by_region': dict(stats['region_counts'])
        },
        'token_statistics': {
            'by_period': dict(stats['token_counts']),
            'by_region': dict(stats['tokens_by_region']),
            'no_stopwords': {
                'by_period': dict(stats['token_counts_no_stopwords']),
                'by_region': dict(stats['tokens_by_region_no_stopwords'])
            }
        },
        'lexical_diversity': {
            'by_region': {region: np.mean(values) for region, values in stats['lexical_diversity'].items()},
            'by_period': {period: np.mean(values) for period, values in stats['lexical_diversity_by_period'].items()}
        },
        'parallel_text_analysis': {
            'pair_count': len(stats['parallel_texts']),
            'average_similarity': np.mean(stats['parallel_metrics']['similarity_scores']) if stats['parallel_metrics']['similarity_scores'] else 0,
            'average_length_ratio': np.mean(stats['parallel_metrics']['length_ratios']) if stats['parallel_metrics']['length_ratios'] else 0,
            'average_shared_vocabulary': np.mean(stats['parallel_metrics']['shared_vocabulary_ratios']) if stats['parallel_metrics']['shared_vocabulary_ratios'] else 0,
            'similarity_distribution': {
                'min': np.min(stats['parallel_metrics']['similarity_scores']) if stats['parallel_metrics']['similarity_scores'] else 0,
                'max': np.max(stats['parallel_metrics']['similarity_scores']) if stats['parallel_metrics']['similarity_scores'] else 0,
                'std': np.std(stats['parallel_metrics']['similarity_scores']) if stats['parallel_metrics']['similarity_scores'] else 0
            }
        }
    }
    
    # Save enhanced corpus stats
    stats_path = base_output_path / "md_out" / "enhanced_corpus_stats.yaml"
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(corpus_stats, f, allow_unicode=True)

###############################################################
#        Parallel texts added for additional data analysis    #
###############################################################

def extract_parallel_texts(text: str) -> list:
    """Extract parallel texts based on formatting patterns."""
    parallel_pairs = []
    lines = text.split('\n')
    current_original = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # If we find an underscored text, it's likely a translation
        if line.startswith('_') and line.endswith('_'):
            # Look at previous non-empty lines for original text
            if current_original:
                original_text = '\n'.join(current_original).strip()
                translation = line.strip('_').strip()
                
                # Analyze the pair
                pair_analysis = analyze_parallel_text_pair(original_text, translation)
                if pair_analysis['pair_metrics']['cosine_similarity'] > 0.3:  # Minimum similarity threshold
                    parallel_pairs.append({
                        'original': original_text,
                        'translation': translation,
                        'metrics': pair_analysis['pair_metrics'],
                        'similarity_score': pair_analysis['pair_metrics']['cosine_similarity']
                    })
                current_original = []
        else:
            current_original.append(line)
    
    return parallel_pairs

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
    verified_pairs = []
    for pair in pairs:
        original, translation = pair['original'], pair['translation']
        
        # Normalize both texts using the character mapping
        normalized_original = normalize_text(original)
        normalized_translation = normalize_text(translation)
        
        similarity = calculate_similarity(normalized_original, normalized_translation)
        if similarity >= similarity_threshold:
            # Calculate length ratio
            orig_tokens = word_tokenize(original)
            trans_tokens = word_tokenize(translation)
            length_ratio = len(trans_tokens) / len(orig_tokens) if orig_tokens else 0
            
            verified_pairs.append({
                'original': original,
                'translation': translation,
                'similarity_score': similarity,
                'pair_metrics': {
                    'length_ratio': length_ratio,
                    'similarity': similarity
                }
            })

    return verified_pairs

def analyze_text_complexity(text: str) -> dict:
    """Analyze text complexity including morphological features."""
    return {
        'morphological_complexity': calculate_morphological_complexity([text]),
        'average_word_length': np.mean([len(word) for word in word_tokenize(text)]),
        'suffix_distribution': analyze_suffix_distribution(text)
    }

def analyze_suffix_distribution(text: str) -> dict:
    """Analyze distribution of different suffix types."""
    normalized_text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(normalized_text)
    
    suffix_counts = defaultdict(int)
    for word in words:
        for category, suffixes in OTTOMAN_SUFFIXES.items():
            for suffix in suffixes:
                if word.endswith(suffix):
                    suffix_counts[category] += 1
    
    return dict(suffix_counts)

def pdf_to_markdown(pdf_file: pathlib.Path, stats: dict, semantic: bool = False) -> bool:
    """Process PDF file and extract text features."""
    try:
        text = pymupdf4llm.extract_text(str(pdf_file))
        if not text.strip():
            return False
            
        # Extract metadata and region
        metadata = extract_metadata(pdf_file)
        region = pdf_file.parent.name
        
        # Calculate text complexity metrics
        complexity_analysis = analyze_text_complexity(text)
        
        # Update stats with complexity metrics
        if 'morphological_analysis' not in stats:
            stats['morphological_analysis'] = defaultdict(list)
        stats['morphological_analysis'][region].append(complexity_analysis['morphological_complexity'])
        
        # Update metadata with complexity metrics
        metadata['statistics']['text_complexity'] = {
            'morphological_complexity': float(complexity_analysis['morphological_complexity']),
            'average_word_length': float(complexity_analysis['average_word_length']),
            'suffix_distribution': complexity_analysis['suffix_distribution']
        }
        
        # Create output structure and save
        output_dir = create_output_structure(pdf_file)
        with open(output_dir / 'metadata.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True)
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        traceback.print_exc()
        return False

def create_enhanced_corpus_stats(stats: dict, base_output_path: pathlib.Path) -> None:
    """Create enhanced corpus statistics including morphological analysis."""
    corpus_stats = {
        'document_counts': {
            'by_period': dict(stats['period_counts']),
            'by_region': dict(stats['region_counts'])
        },
        'token_statistics': {
            'by_period': dict(stats['token_counts']),
            'by_region': dict(stats['tokens_by_region']),
            'no_stopwords': {
                'by_period': dict(stats['token_counts_no_stopwords']),
                'by_region': dict(stats['tokens_by_region_no_stopwords'])
            }
        },
        'morphological_analysis': {
            'by_region': {
                region: float(np.mean(values)) 
                for region, values in stats.get('morphological_analysis', {}).items()
            }
        },
        'lexical_diversity': {
            'by_region': {
                region: np.mean(values) 
                for region, values in stats['lexical_diversity'].items()
            },
            'by_period': {
                period: np.mean(values) 
                for period, values in stats['lexical_diversity_by_period'].items()
            }
        }
    }
    
    # Save enhanced corpus stats
    stats_path = base_output_path / "md_out" / "enhanced_corpus_stats.yaml"
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(corpus_stats, f, allow_unicode=True)

# Add this to the visualizer class
def plot_morphological_complexity(self):
    import matplotlib.pyplot as plt
    import seaborn as sns
    """Create visualization for morphological complexity across regions."""
    plt.figure(figsize=(12, 6))
    
    regions = list(self.stats['morphological_analysis']['by_region'].keys())
    complexity_scores = [self.stats['morphological_analysis']['by_region'][r] for r in regions]
    
    sns.barplot(x=regions, y=complexity_scores)
    plt.title('Morphological Complexity by Region')
    plt.xlabel('Region')
    plt.ylabel('Complexity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('morphological_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()


def pdf_to_markdown(pdf_file: pathlib.Path, stats: dict, semantic: bool = False) -> bool:
    """Process PDF file and extract parallel texts."""
    try:
        # Extract text
        text = pymupdf4llm.extract_text(str(pdf_file))
        if not text.strip():
            return False
            
        # Extract metadata
        metadata = create_metadata(pdf_file)
        
        # Extract parallel texts
        parallel_pairs = extract_parallel_texts(text)
        
        # Update stats with parallel text information
        if parallel_pairs:
            stats['parallel_texts'].extend(parallel_pairs)
            for pair in parallel_pairs:
                stats['parallel_metrics']['similarity_scores'].append(pair['similarity_score'])
                stats['parallel_metrics']['length_ratios'].append(pair['metrics']['length_ratio'])
                stats['parallel_metrics']['shared_vocabulary_ratios'].append(pair['metrics']['shared_vocabulary_ratio'])
        
        # Create output directory structure
        output_dir = create_output_structure(pdf_file)
        
        # Save parallel texts if found
        if parallel_pairs:
            save_parallel_texts(parallel_pairs, output_dir)
            
        # Save metadata with parallel text information
        metadata['statistics']['parallel_texts'] = {
            'count': len(parallel_pairs),
            'average_similarity': float(np.mean([p['similarity_score'] for p in parallel_pairs])) if parallel_pairs else 0
        }
        
        with open(output_dir / 'metadata.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True)
            
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        traceback.print_exc()
        return False

def normalize_text(text):
    """Normalize text using character mappings."""
    # You'll need to define char_setup dictionary
    char_setup = {}  # Add your character mappings here
    for old, new in char_setup.items():
        text = text.replace(old, new)
    return text


def is_poetry(text):
    """Detect if text is poetry based on common patterns."""
    # Check for common poetry markers
    poetry_markers = [
        'Müfteilün', 'Fâilün', 'Mefâilün', 'Feilâtün',  # Common aruz meters
        '###',  # Header markers for verses
        '\n\n',  # Multiple line breaks between stanzas
    ]
    
    return any(marker in text for marker in poetry_markers)


import numpy as np
import re
from nltk.tokenize import word_tokenize

def calculate_morphological_complexity(texts):
    """Calculate morphological complexity of Ottoman Turkish texts."""
    complexity_scores = []
    
    for text in texts:
        # Normalize text to lowercase and remove punctuation
        normalized_text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(normalized_text)
        if not words:
            continue
            
        # Count morphological markers per word
        morpheme_counts = []
        for word in words:
            count = 0
            for category, suffix_list in OTTOMAN_SUFFIXES.items():
                for suffix in suffix_list:
                    if word.endswith(suffix):
                        count += 1
            morpheme_counts.append(count)
        
        if morpheme_counts:
            complexity_scores.append(np.mean(morpheme_counts))
    
    return np.mean(complexity_scores) if complexity_scores else 0.0


def calculate_punctuation_density(texts):
    """Calculate punctuation density of texts."""
    punctuation_marks = set('.,!?;:""''()[]{}«»-–—')
    
    densities = []
    for text in texts:
        total_chars = len(text)
        if total_chars == 0:
            continue
            
        punct_count = sum(1 for char in text if char in punctuation_marks)
        densities.append(punct_count / total_chars)
    
    return {
        'average_density': np.mean(densities) if densities else 0.0,
        'punctuation_distribution': Counter(char for text in texts 
                                         for char in text if char in punctuation_marks)
    }

def calculate_register_score(words):
    """Calculate formality register score."""
    formal_markers = set(REGISTER_MARKERS['formal_markers'])
    informal_markers = set(REGISTER_MARKERS['informal_markers'])
    
    # Count the occurrences of formal and informal markers
    formal_count = sum(1 for word in words if word.lower() in formal_markers)
    informal_count = sum(1 for word in words if word.lower() in informal_markers)
    
    total = formal_count + informal_count
    if total == 0:
        return 0.5  # Neutral score
    
    return formal_count / total


def identify_formality_markers(words):
    """Identify markers of formality in text."""
    formality_markers = []
    
    # Formal constructions
    formal_patterns = {
        'buyurmak': 1.0,
        'teşrif': 1.0,
        'istirham': 1.0,
        'bendeniz': 1.0,
        'zat-ı': 1.0
    }
    
    # Check for formal patterns
    for word in words:
        word_lower = word.lower()
        for pattern, score in formal_patterns.items():
            if pattern in word_lower:
                formality_markers.append((word, score))
    
    return formality_markers

def identify_style_markers(texts, text_type):
    """Identify style markers in texts."""
    markers_found = defaultdict(int)
    
    for text in texts:
        words = word_tokenize(text.lower())
        
        # Count style markers using the loaded JSON data
        for category, markers in STYLE_MARKERS.items():
            if isinstance(markers, dict):
                for subcategory, submarkers in markers.items():
                    for marker in submarkers:
                        markers_found[f"{category}_{subcategory}"] += sum(1 for word in words if marker in word.lower())
            else:
                for marker in markers:
                    markers_found[category] += sum(1 for word in words if marker in word.lower())
    
    return dict(markers_found)

def analyze_text_features(texts, text_type):
    """Analyze linguistic features of texts."""
    try:
        words = [word for text in texts for word in word_tokenize(text.lower())]
        word_counts = Counter(words)
        
        return {
            'vocabulary': {
                'total_tokens': len(words),
                'unique_tokens': len(set(words)),
                'hapax_legomena': len([w for w, c in word_counts.items() if c == 1]),
                'average_word_length': np.mean([len(w) for w in words]),
                'most_frequent_words': dict(word_counts.most_common(20))
            },
            'structural': {
                'average_sentence_length': np.mean([len(sent.split()) 
                    for text in texts for sent in text.split('.')]),
                'morphological_complexity': calculate_morphological_complexity(texts),
                'punctuation_density': calculate_punctuation_density(texts)
            },
            'style_markers': identify_style_markers(texts, text_type)
        }
    except Exception as e:
        print(f"Error in text feature analysis: {e}")
        return {
            'vocabulary': {'total_tokens': 0, 'unique_tokens': 0, 'hapax_legomena': 0,
                         'average_word_length': 0, 'most_frequent_words': {}},
            'structural': {'average_sentence_length': 0, 'morphological_complexity': 0,
                         'punctuation_density': 0},
            'style_markers': {}
        }
    

def analyze_parallel_texts(parallel_pairs):
    """Comprehensive analysis of parallel texts."""
    analysis = {
        'overall_statistics': {
            'pair_count': len(parallel_pairs),
            'average_similarity': np.mean([p['similarity_score'] for p in parallel_pairs]),
            'text_types': {
                'poetry': sum(1 for p in parallel_pairs if is_poetry(p['original'])),
                'prose': sum(1 for p in parallel_pairs if not is_poetry(p['original']))
            }
        },
        'linguistic_features': {
            'original': analyze_text_features([p['original'] for p in parallel_pairs], 'original'),
            'modern': analyze_text_features([p['translation'] for p in parallel_pairs], 'modern')
        }
    }
    return analysis

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


def process_directory(input_dir, parallel=False, semantic=False):
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
                'by_region': {},
                'samples': []  # Store actual parallel text samples
            }
        }
        
        # Process files
        processed_count = 0
        for pdf_file in pathlib.Path(input_dir).rglob('*.pdf'):
            try:
                print(f"\nProcessing: {pdf_file}")
                result = pdf_to_markdown(pdf_file, stats, semantic=semantic)
                if result:
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
    if isinstance(obj, (defaultdict, dict)):
        return {k: convert_defaultdict_to_dict(v) for k, v in dict(obj).items()}
    elif isinstance(obj, list):
        return [convert_defaultdict_to_dict(v) for v in obj]
    return obj

def find_translation_pair(pdf_file):
    """Find translation pair for a given PDF file."""
    # Get the stem of the filename (without extension)
    stem = pdf_file.stem
    
    # Look for files with similar names in the same directory
    for potential_pair in pdf_file.parent.glob('*.pdf'):
        if potential_pair != pdf_file and (
            potential_pair.stem.startswith(stem) or 
            stem.startswith(potential_pair.stem) or
            'translation' in potential_pair.stem.lower() or
            'tercüme' in potential_pair.stem.lower()
        ):
            return potential_pair
    return None

if __name__ == "__main__":
    import argparse
    
    description = """
    Thank you for using OttoMiner!
    
    This tool processes Ottoman Turkish PDF files and performs various analyses including:
    - Parallel text processing
    - Semantic analysis
    - Morphological analysis
    - Visualization generation

    Example usage:
    python pdf_to_md.py /path/to/pdfs -p -s -g -m
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
    
    # Only initialize NLTK if we're actually processing files
    if len(sys.argv) > 1 and not sys.argv[1] in ['-h', '--help']:
        initialize_nltk()
        
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