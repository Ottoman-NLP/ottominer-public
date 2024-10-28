import sys
import pathlib
import pymupdf4llm
from llama_index.core import SimpleDirectoryReader
import unicodedata
import re
import yaml
from collections import defaultdict
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from collections import Counter
import pandas as pd
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import nltk

# Download required NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_parallel_texts(text):
    """Extract original and modern Turkish pairs from text with improved handling."""
    paragraphs = text.split('\n\n')
    parallel_pairs = []

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # Case 1: Direct parallel pairs (original followed by italicized translation)
        if i < len(paragraphs) - 1 and paragraphs[i+1].strip().startswith('_') and paragraphs[i+1].strip().endswith('_'):
            original = para
            translation = paragraphs[i+1].strip('_').strip()
            if is_valid_pair(original, translation):
                parallel_pairs.append({
                    'original': original,
                    'translation': translation,
                    'type': 'direct_pair'
                })

        # Case 2: Poetic stanzas
        elif para.count('\n') > 0:  # Multiple lines indicate possible poetic stanza
            stanza_pairs = extract_poetic_pairs(para)
            if stanza_pairs:
                parallel_pairs.extend(stanza_pairs)

    return parallel_pairs

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
            verified_pairs.append({
                'original': original,
                'translation': translation,
                'similarity_score': similarity
            })

    return verified_pairs

def normalize_text(text):
    """Normalize text using character mappings."""
    for old, new in char_setup.items():
        text = text.replace(old, new)
    return text

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
        },
        'diachronic_analysis': analyze_diachronic_changes(parallel_pairs),
        'dialectal_features': analyze_dialectal_features(parallel_pairs),
        'stylistic_analysis': analyze_style(parallel_pairs)
    }
    return analysis

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

def pdf_to_markdown(pdf_path, stats):
    try:
        base_output_path = pathlib.Path(r"C:\Users\Administrator\Desktop\cook\Ottoman-NLP\corpus-texts")
        pdf_name = pdf_path.stem
        region = pdf_path.parent.name

        # Ensure base output directory exists
        output_base = base_output_path / "md_out"
        output_base.mkdir(parents=True, exist_ok=True)

        # Create region-specific directory
        output_folder = output_base / region / pdf_name
        output_folder.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

    # Convert PDF to markdown
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=False,
        dpi=300,
        force_text=True,
        margins=(50, 50, 0, 0),
        page_chunks=True,
        table_strategy="lines_strict",
        fontsize_limit=5,
        ignore_code=True,
        extract_words=False,
        show_progress=True,
        page_width=842, page_height=1190
    )

    # Process markdown text and extract parallel texts
    if isinstance(md_text, list):
        md_text = clean_markdown_chunks(md_text)

    # Calculate text statistics
    words = word_tokenize(md_text.lower())
    text_stats = calculate_text_stats(md_text)

    # Create metadata BEFORE parallel text processing
    metadata = create_metadata(pdf_path, region, text_stats)

    # Extract and verify parallel texts
    parallel_pairs = extract_parallel_texts(md_text)
    verified_pairs = verify_parallel_pairs(parallel_pairs)

    if verified_pairs:
        parallel_analysis = analyze_parallel_texts(verified_pairs)
        metadata['parallel_texts'] = parallel_analysis
        save_parallel_texts(verified_pairs, output_folder)

    # Create and save metadata with statistics
    metadata_path = output_folder / "metadata.yaml"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, allow_unicode=True)

    # Save markdown
    output_path = output_folder / "original.md"
    output_path.write_text(md_text, encoding='utf-8')

    # Update statistics
    if metadata['period'] != 'Unknown':
        update_statistics(stats, metadata, region, text_stats)

    return output_path

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
    """Update corpus statistics based on analyzed text."""
    stats['period_counts'][metadata['period']] += 1
    stats['region_counts'][region] += 1

    # Update lexical diversity metrics
    stats['lexical_diversity'][region].append(text_stats['lexical_diversity'])
    stats['lexical_diversity_no_stopwords'][region].append(text_stats['lexical_diversity_no_stopwords'])
    stats['lexical_diversity_by_period'][metadata['period']].append(text_stats['lexical_diversity'])
    stats['lexical_diversity_no_stopwords_by_period'][metadata['period']].append(text_stats['lexical_diversity_no_stopwords'])

    # Update token counts
    stats['token_counts'][metadata['period']] += text_stats['token_count']
    stats['token_counts_no_stopwords'][metadata['period']] += text_stats['token_count_no_stopwords']
    stats['tokens_by_region'][region] += text_stats['token_count']
    stats['tokens_by_region_no_stopwords'][region] += text_stats['token_count_no_stopwords']

    # Update stopwords ratio
    stats['stopwords_ratios'][region].append(text_stats['stopwords_ratio'])

# Utilizing multiprocessing for performance optimization
def parallel_analyze_texts(texts):
    import multiprocessing
    """Parallel analysis of text features for performance optimization."""
    with multiprocessing.Pool() as pool:
        results = pool.map(analyze_text_features, texts)
    return results
def analyze_word_order_changes(parallel_pairs):
    """Analyze changes in word order between versions."""
    order_patterns = {
        'sov_to_svo': 0,
        'preserved_order': 0,
        'total_sentences': 0
    }
    
    for pair in parallel_pairs:
        orig_sentences = sent_tokenize(pair['original'])
        trans_sentences = sent_tokenize(pair['translation'])
        
        order_patterns['total_sentences'] += len(orig_sentences)
        
        for orig_sent, trans_sent in zip(orig_sentences, trans_sentences):
            orig_order = analyze_sentence_structure(orig_sent)
            trans_order = analyze_sentence_structure(trans_sent)
            
            if orig_order == trans_order:
                order_patterns['preserved_order'] += 1
            elif orig_order == 'SOV' and trans_order == 'SVO':
                order_patterns['sov_to_svo'] += 1
    
    return order_patterns

def determine_period(year_str):
    """Determine the period, including folkloric texts."""
    try:
        # Convert year_str to string if it's an integer
        year_str = str(year_str)
        
        # Check for folkloric texts (ending with '00')
        if year_str.endswith('00'):
            base_year = int(year_str[:-2])
            return f"Folkloric-{determine_period(base_year)}"
        
        # Regular period determination
        year = int(year_str)
        if year < 1600:
            return "Early Ottoman"
        elif 1600 <= year <= 1800:
            return "Classical Ottoman"
        else:
            return "Late Ottoman"
    except (ValueError, TypeError):
        return "Unknown"
CHAR_NORMALIZATION_TABLE = str.maketrans('', '', string.punctuation)

char_setup = {

    'â': 'a', 'Â': 'A','Â': 'A',
    "å": "a", "Å": "A",
    'ā': 'a', 'Ā': 'A',
    
    "ê": "e",
    "ē": "e", "Ē": "E",

    'î': 'i', 'Î': 'I',
    "ī": "i", "Ī": "I", 

    'û': 'u', 'Û': 'U',

    "ô": "ö",
    "ō": "o",  "Ō": "Ö",
    
}
# Define Ottoman Turkish stopwords
OTTOMAN_STOPWORDS = {
    # Common particles, conjunctions, and function words
    've', 'ile', 'ki', 'bu', 'şu', 'o', 'da', 'de', 'mi', 'mu', 'mı', 'mü', 
    'ise', 'idi', 'imiş', 'olan', 'olarak', 'gibi', 'için', 'üzere', 'kadar', 
    'dahi', 'bile', 'hem', 'amma', 'fakat', 'lakin', 'ancak', 'yalnız', 'yani', 
    'ama', 'ya', 'işte', 'şayet', 'keza', 'çünkü', 'zira', 'hatta', 'veya', 'yahut', 
    'eger', 'eğer', 'değil', 'velev', 'beraber', 'sanki', 'zaten', 'lâkin', 'belki', 
    'şöyle', 'şöylece', 'artık', 'neden', 'niçin', 'şayet', 'ancak', 'fekat', 'hala',
    
    # Articles, demonstratives, and determiners
    'bir', 'şol', 'işbu', 'ol', 'bütün', 'her', 'hiçbir', 'bazı', 'bazısı', 'kendi', 
    'herhangi', 'kim', 'hepsi', 'çoğu', 'tüm', 'diğer', 'diğeri', 'bazıları', 
    
    # Pronouns
    'ben', 'sen', 'biz', 'siz', 'o', 'şu', 'bu', 'onlar', 'şunlar', 'bunlar', 'hepsi', 
    'kim', 'herkes', 'kendi', 'kendisi', 'biri', 'birisi', 'birkaç', 
    
    # Prepositions
    'göre', 'doğru', 'karşı', 'rağmen', 'nazaran', 'üzerine', 'içinde', 'içinden', 
    'sonra', 'önce', 'altında', 'üstünde', 'arasında', 'yanında', 'içeri', 'dışarı', 
    'hakkında', 'şayet', 'boyunca', 'yanı', 'arada', 'önünden', 'arkasından', 'tarafından', 
    
    # Common auxiliary verbs
    'olmak', 'etmek', 'eylemek', 'kılmak', 'bulunmak', 'demek', 'yapmak', 'gitmek', 
    'gelmek', 'durmak', 'varmak', 'sanmak', 'bilmek', 'anlamak', 'saymak', 'düşmek',
    
    # Suffixes (when tokenized separately)
    'dir', 'dır', 'dur', 'dür', 'tir', 'tır', 'tur', 'tür', 'miş', 'mişçesine', 'dık', 
    'dik', 'duk', 'dük', 'tır', 'tıran', 'tırken', 'dıktan', 'tıkça', 'tığı', 'liği',
    
    # Numbers and numerical terms
    'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on', 'yüz', 
    'bin', 'milyon', 'milyar', 'ilk', 'son', 'kaç', 'birkaç', 'çok', 'az', 'birçok', 
    'kaçıncı', 'üçüncü', 'beşinci', 'ilk', 'sonra', 'önce', 'defa', 'kere', 'kez'
}


def calculate_lexical_diversity(text, use_stopwords=True):
    """Calculate lexical diversity with option to remove stopwords."""
    words = word_tokenize(text.lower())
    if use_stopwords:
        words = [word for word in words if word not in OTTOMAN_STOPWORDS]
    unique_words = set(words)
    if len(words) == 0:
        return 0
    return len(unique_words) / len(words)

def calculate_text_stats(text):
    """Calculate comprehensive text statistics."""
    # Normalize text
    for old, new in char_setup.items():
        text = text.replace(old, new)
    
    # Remove punctuation
    text = text.translate(CHAR_NORMALIZATION_TABLE)
    
    words = word_tokenize(text.lower())
    words_no_stop = [word for word in words if word not in OTTOMAN_STOPWORDS]
    
    stats = {
        'token_count': len(words),
        'token_count_no_stopwords': len(words_no_stop),
        'unique_tokens': len(set(words)),
        'unique_tokens_no_stopwords': len(set(words_no_stop)),
        'lexical_diversity': len(set(words)) / len(words) if words else 0,
        'lexical_diversity_no_stopwords': len(set(words_no_stop)) / len(words_no_stop) if words_no_stop else 0,
        'stopwords_ratio': (len(words) - len(words_no_stop)) / len(words) if words else 0
    }
    return stats

def create_metadata(pdf_path, region, text_stats):
    """Create metadata with error handling."""
    # Extract year from filename
    year_match = re.search(r'_(\d{4})\.pdf$', str(pdf_path))
    year = int(year_match.group(1)) if year_match else None
    
    metadata = {
        'filename': pdf_path.name,
        'date': year,
        'period': determine_period(year) if year is not None else 'Unknown',
        'region': region,
        'author': '',  # Left empty for manual filling
        'title': '',   # Left empty for manual filling
        'language': 'Ottoman Turkish',
        'genre': '',   # Left empty for manual filling
        'statistics': {
            'token_count': text_stats['token_count'],
            'unique_tokens': text_stats['unique_tokens'],
            'lexical_diversity': text_stats['lexical_diversity']
        }
    }
    return metadata

def create_corpus_stats(stats, base_output_path):
    """Create corpus-wide statistics with proper error handling."""
    try:
        # Filter out empty lists and calculate averages safely
        corpus_stats = {
            'period_distribution': dict(stats['period_counts']),
            'region_distribution': dict(stats['region_counts']),
            'corpus_wide_statistics': {
                'average_document_length': (
                    sum(stats['token_counts'].values()) / sum(stats['period_counts'].values())
                    if sum(stats['period_counts'].values()) > 0 else 0
                ),
                'average_document_length_no_stopwords': (
                    sum(stats['token_counts_no_stopwords'].values()) / sum(stats['period_counts'].values())
                    if sum(stats['period_counts'].values()) > 0 else 0
                ),
                'average_lexical_diversity': (
                    sum(sum(diversities) for diversities in stats['lexical_diversity'].values() if diversities) /
                    sum(len(diversities) for diversities in stats['lexical_diversity'].values() if diversities)
                    if any(stats['lexical_diversity'].values()) else 0
                ),
                'average_lexical_diversity_no_stopwords': (
                    sum(sum(diversities) for diversities in stats['lexical_diversity_no_stopwords'].values() if diversities) /
                    sum(len(diversities) for diversities in stats['lexical_diversity_no_stopwords'].values() if diversities)
                    if any(stats['lexical_diversity_no_stopwords'].values()) else 0
                ),
                'average_stopwords_ratio': (
                    sum(sum(ratios) for ratios in stats['stopwords_ratios'].values() if ratios) /
                    sum(len(ratios) for ratios in stats['stopwords_ratios'].values() if ratios)
                    if any(stats['stopwords_ratios'].values()) else 0
                )
            }
        }

        # Ensure output directory exists
        stats_path = base_output_path / "md_out"
        stats_path.mkdir(parents=True, exist_ok=True)
        
        # Save corpus stats
        with open(stats_path / "corpus_stats.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(corpus_stats, f, allow_unicode=True)
            
    except Exception as e:
        print(f"Error creating corpus statistics: {e}")
        traceback.print_exc()

def process_directory(input_dir):
    try:
        stats = {
            'period_counts': defaultdict(int),
            'region_counts': defaultdict(int),
            'token_counts': defaultdict(int),
            'token_counts_no_stopwords': defaultdict(int),
            'tokens_by_region': defaultdict(int),
            'tokens_by_region_no_stopwords': defaultdict(int),
            'lexical_diversity': defaultdict(list),
            'lexical_diversity_no_stopwords': defaultdict(list),
            'lexical_diversity_by_period': defaultdict(list),
            'lexical_diversity_no_stopwords_by_period': defaultdict(list),
            'stopwords_ratios': defaultdict(list)
        }

        input_path = pathlib.Path(input_dir)
        base_output_path = pathlib.Path(r"C:\Users\Administrator\Desktop\cook\Ottoman-NLP\corpus-texts")
        output_dir = base_output_path / "md_out"
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory created at: {output_dir}")

        # Process files...
        
        return stats
        
    except Exception as e:
        print(f"Error in process_directory: {e}")
        traceback.print_exc()
        return None
def extract_text_statistics(pdf_path):
    """Extract text statistics from PDF file."""
    try:
        # Extract text from PDF using pymupdf4llm correctly
        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            write_images=False,
            force_text=True,
            margins=(50, 50, 0, 0),
            page_chunks=True,
            table_strategy="lines_strict",
            fontsize_limit=5,
            ignore_code=True,
            extract_words=False
        )
        
        # Handle the returned markdown text
        if isinstance(md_text, list):
            text = '\n'.join(chunk.get('text', '') for chunk in md_text if 'text' in chunk)
        else:
            text = md_text
        
        # Tokenize text
        tokens = word_tokenize(text)
        tokens_no_stopwords = [t for t in tokens if t.lower() not in OTTOMAN_STOPWORDS]
        
        # Calculate statistics
        stats = {
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            'token_count_no_stopwords': len(tokens_no_stopwords),
            'unique_tokens_no_stopwords': len(set(tokens_no_stopwords)),
            'lexical_diversity_no_stopwords': len(set(tokens_no_stopwords)) / len(tokens_no_stopwords) if tokens_no_stopwords else 0
        }
        
        return stats
        
    except Exception as e:
        print(f"Error extracting text statistics: {e}")
        return {
            'token_count': 0,
            'unique_tokens': 0,
            'lexical_diversity': 0,
            'token_count_no_stopwords': 0,
            'unique_tokens_no_stopwords': 0,
            'lexical_diversity_no_stopwords': 0
        }
    
def print_detailed_report(stats):
    """Print a detailed statistical report."""
    print("\nPeriod Distribution:")
    for period, count in stats['period_counts'].items():
        print(f"\n{period}:")
        print(f"  Texts: {count}")
        print(f"  Tokens: {stats['token_counts'][period]}")
        
        if stats['lexical_diversity_by_period'][period]:
            avg_diversity = sum(stats['lexical_diversity_by_period'][period]) / len(stats['lexical_diversity_by_period'][period])
            print(f"  Average lexical diversity: {avg_diversity:.3f}")
            
        if stats['semantic_changes'][period]:
            avg_semantic = sum(stats['semantic_changes'][period]) / len(stats['semantic_changes'][period])
            print(f"  Average semantic preservation: {avg_semantic:.3f}")
            
        if stats['rhyme_patterns'][period]:
            avg_rhyme = sum(stats['rhyme_patterns'][period]) / len(stats['rhyme_patterns'][period])
            print(f"  Average rhyme preservation: {avg_rhyme:.3f}")

    print("\nRegional Distribution:")
    for region, count in stats['region_counts'].items():
        print(f"\n{region}:")
        print(f"  Texts: {count}")
        print(f"  Tokens: {stats['tokens_by_region'][region]}")
        
        if stats['lexical_diversity'][region]:
            avg_diversity = sum(stats['lexical_diversity'][region]) / len(stats['lexical_diversity'][region])
            print(f"  Average lexical diversity: {avg_diversity:.3f}")
            
        if stats['dialect_features'][region]:
            avg_dialect = sum(stats['dialect_features'][region]) / len(stats['dialect_features'][region])
            print(f"  Average dialect confidence: {avg_dialect:.3f}")

def analyze_parallel_text_pair(original, translation):
    """Analyze linguistic features of parallel text pairs."""
    # Normalize both texts
    normalized_original = original
    normalized_translation = translation
    for old, new in char_setup.items():
        normalized_original = normalized_original.replace(old, new)
        normalized_translation = normalized_translation.replace(old, new)
    
    # Calculate statistics
    orig_tokens = word_tokenize(normalized_original)
    trans_tokens = word_tokenize(normalized_translation)
    
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
            'similarity_score': calculate_similarity(normalized_original, normalized_translation),
            'shared_vocabulary': len(set(orig_tokens) & set(trans_tokens)),
        }
    }


def extract_poetic_pairs(text):
    """Extract parallel pairs from poetic text."""
    lines = text.split('\n')
    pairs = []
    current_original = []
    current_translation = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('_') and line.endswith('_'):
            if current_original:  # We have a complete pair
                orig_text = '\n'.join(current_original)
                trans_text = line.strip('_')
                if is_valid_pair(orig_text, trans_text):
                    pairs.append({
                        'original': orig_text,
                        'translation': trans_text,
                        'type': 'poetic_pair'
                    })
                current_original = []
            current_translation = []
        else:
            current_original.append(line)
    
    return pairs

def is_valid_pair(original, translation):
    """Validate if the pair is legitimate."""
    if not original or not translation:
        return False
        
    # Basic validation
    if len(original.split()) < 2 or len(translation.split()) < 2:
        return False
        
    # Calculate similarity
    similarity = calculate_similarity(original, translation)
    return similarity >= 0.2  # Adjust threshold as needed

def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts."""
    try:
        # Clean and normalize texts
        text1 = clean_text(text1)
        text2 = clean_text(text2)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=OTTOMAN_STOPWORDS,
            ngram_range=(1, 2)
        )
        
        # Transform texts to vectors
        vectors = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def clean_text(text):
    """Clean and normalize text."""
    try:
        # Convert to string if not already
        text = str(text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Apply character normalization
        for old, new in char_setup.items():
            text = text.replace(old, new)
        
        # Remove punctuation
        text = text.translate(CHAR_NORMALIZATION_TABLE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.lower()
        
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

def is_poetry(text):
    """Detect if text is poetry based on structure and features."""
    lines = text.split('\n')
    
    # Poetry indicators
    indicators = {
        'rhyme_ratio': 0,
        'regular_length': 0,
        'line_count': len(lines),
        'meter_matches': 0
    }
    
    # Common Ottoman poetry meters
    meters = {
        'aruz': [11, 14, 15, 16],  # Common aruz syllable counts
        'hece': [7, 8, 11, 14]     # Common hece vezni syllable counts
    }
    
    def count_syllables(line):
        """Count syllables in Ottoman Turkish line."""
        vowels = set('aâeêıîiîoôöuûü')
        count = 0
        prev_char = ''
        
        for char in line.lower():
            if char in vowels and prev_char not in vowels:
                count += 1
            prev_char = char
        return count
    
    # Analyze each line
    line_lengths = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check syllable count
        syllables = count_syllables(line)
        line_lengths.append(syllables)
        
        # Check if matches common meters
        if syllables in meters['aruz'] or syllables in meters['hece']:
            indicators['meter_matches'] += 1
    
    # Calculate regularity of line lengths
    if line_lengths:
        length_counts = Counter(line_lengths)
        most_common_length = length_counts.most_common(1)[0][1]
        indicators['regular_length'] = most_common_length / len(line_lengths)
    
    # Determine if it's poetry
    return (indicators['regular_length'] > 0.6 or 
            indicators['meter_matches'] / indicators['line_count'] > 0.5)

def has_religious_content(text):
    """Detect religious content in text."""
    religious_terms = {
        'islamic': [
            'allah', 'bismillah', 'elhamdülillah', 'inşallah',
            'cami', 'namaz', 'oruç', 'hac', 'zekat',
            'mübarek', 'helal', 'haram', 'sevap', 'günah',
            'peygamber', 'hadis', 'ayet', 'sure', 'kuran'
        ],
        'honorifics': [
            'hazret', 'efendi', 'şeyh', 'molla', 'imam',
            'müftü', 'derviş', 'veli', 'evliya'
        ]
    }
    
    text = text.lower()
    term_count = 0
    
    for category in religious_terms.values():
        for term in category:
            term_count += text.count(term)
    
    # Consider it religious if density of religious terms is high enough
    return term_count / len(text.split()) > 0.02

def has_folk_elements(text):
    """Detect folk literature elements."""
    folk_markers = {
        'genres': [
            'masal', 'destan', 'mani', 'türkü', 'ninni',
            'atasözü', 'bilmece', 'tekerleme', 'efsane'
        ],
        'formulas': [
            'bir varmış bir yokmuş', 'evvel zaman içinde',
            'az gitmiş uz gitmiş', 'gülü gülüstanı'
        ],
        'characters': [
            'keloğlan', 'dev', 'peri', 'cadı', 'padişah',
            'derviş', 'köylü', 'çoban'
        ]
    }
    
    text = text.lower()
    marker_count = 0
    
    for category in folk_markers.values():
        for marker in category:
            marker_count += text.count(marker)
    
    return marker_count > 0

def calculate_length_stats(texts):
    """Calculate length statistics for texts."""
    if not texts:
        return {
            'average_length': 0,
            'std_dev': 0,
            'min_length': 0,
            'max_length': 0,
            'total_words': 0
        }
    
    lengths = [len(text.split()) for text in texts]
    
    return {
        'average_length': np.mean(lengths),
        'std_dev': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'total_words': sum(lengths)
    }

def calculate_vocabulary_stats(parallel_pairs):
    """Calculate vocabulary statistics for parallel texts."""
    orig_words = []
    trans_words = []
    
    for pair in parallel_pairs:
        orig_words.extend(word_tokenize(pair['original'].lower()))
        trans_words.extend(word_tokenize(pair['translation'].lower()))
    
    orig_vocab = Counter(orig_words)
    trans_vocab = Counter(trans_words)
    
    return {
        'original': {
            'unique_words': len(orig_vocab),
            'total_words': len(orig_words),
            'lexical_diversity': len(orig_vocab) / len(orig_words) if orig_words else 0,
            'most_common': dict(orig_vocab.most_common(20))
        },
        'translation': {
            'unique_words': len(trans_vocab),
            'total_words': len(trans_words),
            'lexical_diversity': len(trans_vocab) / len(trans_words) if trans_words else 0,
            'most_common': dict(trans_vocab.most_common(20))
        },
        'comparison': {
            'vocabulary_ratio': len(orig_vocab) / len(trans_vocab) if trans_vocab else 0,
            'shared_vocabulary': len(set(orig_vocab) & set(trans_vocab))
        }
    }

# Required imports at the top of the file
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize

def classify_text_types(parallel_pairs):
    """Classify texts into types (poetry, prose, etc.)."""
    text_types = defaultdict(int)
    
    for pair in parallel_pairs:
        # Check for poetic features
        if is_poetry(pair['original']):
            text_types['poetry'] += 1
        else:
            text_types['prose'] += 1
            
        # Check for specific genres
        if has_religious_content(pair['original']):
            text_types['religious'] += 1
        if has_folk_elements(pair['original']):
            text_types['folk_literature'] += 1
    
    return dict(text_types)

def identify_time_period(parallel_pairs):
    """Identify the time period of texts based on linguistic features."""
    time_markers = {
        'ottoman_classical': ['efendi', 'hazret', 'cenab'],
        'tanzimat': ['mektep', 'muallim', 'terakki'],
        'modern': ['okul', 'öğretmen', 'gelişme']
    }
    
    period_scores = defaultdict(int)
    
    for pair in parallel_pairs:
        text = pair['original'].lower()
        for period, markers in time_markers.items():
            for marker in markers:
                period_scores[period] += text.count(marker)
    
    # Determine predominant period
    if period_scores:
        main_period = max(period_scores.items(), key=lambda x: x[1])[0]
        return {
            'primary_period': main_period,
            'period_distribution': dict(period_scores)
        }
    return {'primary_period': 'unknown', 'period_distribution': {}}
# Define dialect feature constants
NORTHERN_FEATURES = {
    # Phonological features
    'g->k': ['kelgen', 'ketken', 'körgen'],
    'b->p': ['parmak', 'pıçak', 'palta'],
    'vowel_raising': ['kelin', 'kerek', 'keçe']
}

CENTRAL_FEATURES = {
    # Phonological features
    'd->t': ['taş', 'tağ', 'til'],
    'ñ->n': ['ana', 'sona', 'tana'],
    'vowel_harmony': ['keldi', 'berdi', 'kördi']
}

SOUTHERN_FEATURES = {
    # Phonological features
    'ğ->h': ['dağ->dah', 'bağ->bah'],
    'k->h': ['hün', 'höz', 'hel'],
    'vowel_lowering': ['alma', 'bala', 'kara']
}

def identify_dialect_region(parallel_pairs):
    """Identify dialect region based on linguistic features."""
    try:
        dialect_features = analyze_dialectal_features(parallel_pairs)
        
        # Initialize region scores with weights
        region_scores = {
            'northern': 0,
            'central': 0,
            'southern': 0
        }
        
        # Weight factors for different feature types
        weights = {
            'phonological': 2.0,
            'morphological': 1.5,
            'lexical': 1.0
        }
        
        # Analyze phonological features
        for feature_type, features in dialect_features['phonological'].items():
            for feature, count in features.items():
                # Check in northern features
                for pattern_group in NORTHERN_FEATURES.values():
                    if any(pattern in feature for pattern in pattern_group):
                        region_scores['northern'] += count * weights['phonological']
                
                # Check in central features
                for pattern_group in CENTRAL_FEATURES.values():
                    if any(pattern in feature for pattern in pattern_group):
                        region_scores['central'] += count * weights['phonological']
                
                # Check in southern features
                for pattern_group in SOUTHERN_FEATURES.values():
                    if any(pattern in feature for pattern in pattern_group):
                        region_scores['southern'] += count * weights['phonological']
        
        # Analyze morphological features
        morph_features = dialect_features.get('morphological', {})
        for feature, count in morph_features.items():
            if feature in ['gan', 'kan', 'gen', 'ken']:
                region_scores['northern'] += count * weights['morphological']
            elif feature in ['dı', 'di', 'tı', 'ti']:
                region_scores['central'] += count * weights['morphological']
            elif feature in ['ğan', 'ğen']:
                region_scores['southern'] += count * weights['morphological']
        
        # Analyze lexical features
        lexical_features = dialect_features.get('lexical', {}).get('dialect_specific_terms', {})
        for term, info in lexical_features.items():
            if info.get('region') == 'northern':
                region_scores['northern'] += info.get('frequency', 0) * weights['lexical']
            elif info.get('region') == 'central':
                region_scores['central'] += info.get('frequency', 0) * weights['lexical']
            elif info.get('region') == 'southern':
                region_scores['southern'] += info.get('frequency', 0) * weights['lexical']
        
        # Determine primary region
        if any(region_scores.values()):
            primary_region = max(region_scores.items(), key=lambda x: x[1])[0]
            confidence_score = region_scores[primary_region] / sum(region_scores.values())
        else:
            primary_region = 'unknown'
            confidence_score = 0.0
        
        return {
            'primary_region': primary_region,
            'confidence_score': confidence_score,
            'regional_distribution': dict(region_scores),
            'feature_counts': {
                'phonological': len(dialect_features.get('phonological', {})),
                'morphological': len(dialect_features.get('morphological', {})),
                'lexical': len(dialect_features.get('lexical', {}).get('dialect_specific_terms', {}))
            }
        }
        
    except Exception as e:
        print(f"Error in dialect region identification: {e}")
        return {
            'primary_region': 'unknown',
            'confidence_score': 0.0,
            'regional_distribution': {},
            'error': str(e)
        }
    
def identify_dialect_region(parallel_pairs):
    """Identify dialect region based on linguistic features."""
    dialect_features = analyze_dialectal_features(parallel_pairs)
    
    # Analyze regional distribution
    region_scores = defaultdict(int)
    for feature_type, features in dialect_features['phonological'].items():
        for feature, count in features.items():
            if feature in NORTHERN_FEATURES:
                region_scores['northern'] += count
            elif feature in CENTRAL_FEATURES:
                region_scores['central'] += count
            elif feature in SOUTHERN_FEATURES:
                region_scores['southern'] += count
    
    return {
        'primary_region': max(region_scores.items(), key=lambda x: x[1])[0] if region_scores else 'unknown',
        'regional_distribution': dict(region_scores)
    }

def calculate_statistical_measures(parallel_pairs):
    """Calculate statistical measures for parallel texts."""
    measures = {
        'length_statistics': {
            'original': calculate_length_stats([p['original'] for p in parallel_pairs]),
            'translation': calculate_length_stats([p['translation'] for p in parallel_pairs])
        },
        'similarity_scores': {
            'average': np.mean([p.get('similarity_score', 0) for p in parallel_pairs]),
            'std_dev': np.std([p.get('similarity_score', 0) for p in parallel_pairs]),
            'range': (
                min([p.get('similarity_score', 0) for p in parallel_pairs]),
                max([p.get('similarity_score', 0) for p in parallel_pairs])
            )
        },
        'vocabulary_statistics': calculate_vocabulary_stats(parallel_pairs)
    }
    return measures

def suggest_research_applications(parallel_pairs):
    """Suggest potential research applications."""
    suggestions = []
    
    # Analyze text characteristics
    text_types = classify_text_types(parallel_pairs)
    dialect_info = identify_dialect_region(parallel_pairs)
    
    if text_types.get('poetry', 0) > 0:
        suggestions.append({
            'field': 'Literary Studies',
            'focus': 'Poetic Translation Analysis',
            'potential': 'High' if text_types['poetry'] > len(parallel_pairs)/2 else 'Medium'
        })
    
    if dialect_info['primary_region'] != 'unknown':
        suggestions.append({
            'field': 'Dialectology',
            'focus': f"Analysis of {dialect_info['primary_region'].title()} Dialect Features",
            'potential': 'High'
        })
    
    return suggestions

def identify_limitations(parallel_pairs):
    """Identify limitations in the parallel texts."""
    limitations = []
    
    # Check text quality
    avg_similarity = np.mean([p.get('similarity_score', 0) for p in parallel_pairs])
    if avg_similarity < 0.5:
        limitations.append({
            'type': 'translation_quality',
            'description': 'Low average similarity between parallel texts',
            'severity': 'High'
        })
    
    # Check completeness
    missing_translations = sum(1 for p in parallel_pairs if not p.get('translation'))
    if missing_translations > 0:
        limitations.append({
            'type': 'completeness',
            'description': f'Missing translations for {missing_translations} texts',
            'severity': 'Medium' if missing_translations < len(parallel_pairs)/2 else 'High'
        })
    
    return limitations

def save_parallel_formats(parallel_pairs, output_folder):
    import csv
    """Save parallel texts in multiple formats."""
    output_folder = Path(output_folder)
    
    # Save as TSV
    tsv_path = output_folder / "parallel_texts.tsv"
    with open(tsv_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['original', 'translation', 'similarity_score'])
        for pair in parallel_pairs:
            writer.writerow([
                pair['original'],
                pair['translation'],
                pair.get('similarity_score', '')
            ])
    
    # Save as JSON
    json_path = output_folder / "parallel_texts.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parallel_pairs, f, ensure_ascii=False, indent=2)
    
    # Save as aligned text
    aligned_path = output_folder / "parallel_texts_aligned.txt"
    with open(aligned_path, 'w', encoding='utf-8') as f:
        for pair in parallel_pairs:
            f.write(f"ORIGINAL: {pair['original']}\n")
            f.write(f"TRANSLATION: {pair['translation']}\n")
            f.write(f"SIMILARITY: {pair.get('similarity_score', 'N/A')}\n")
            f.write("-" * 80 + "\n")

def save_parallel_texts(parallel_pairs, output_folder):
    """Save parallel texts with enhanced metadata."""
    try:
        if not parallel_pairs:
            print("No parallel pairs to save.")
            return
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive metadata
        metadata = {
            'corpus_information': {
                'total_pairs': len(parallel_pairs),
                'text_types': classify_text_types(parallel_pairs),
                'time_period': identify_time_period(parallel_pairs),
                'dialect_region': identify_dialect_region(parallel_pairs)
            },
            'linguistic_analysis': analyze_parallel_texts(parallel_pairs),
            'statistical_measures': calculate_statistical_measures(parallel_pairs),
            'research_applications': {
                'suggested_uses': suggest_research_applications(parallel_pairs),
                'limitations': identify_limitations(parallel_pairs)
            }
        }
        
        # Save detailed analysis in various formats
        save_analysis_files(metadata, parallel_pairs, output_folder)
        
        # Generate visualization data
        generate_visualization_data(metadata, output_folder)
        
        # Save parallel texts in multiple formats
        save_parallel_formats(parallel_pairs, output_folder)
        
        print(f"Successfully saved parallel texts and analysis to {output_folder}")
        return metadata
        
    except Exception as e:
        print(f"Error saving parallel texts: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def save_parallel_texts(parallel_pairs, output_folder):
    """Save parallel texts with enhanced metadata."""
    if not parallel_pairs:
        return
        
    # Create comprehensive metadata
    metadata = {
        'corpus_information': {
            'total_pairs': len(parallel_pairs),
            'text_types': classify_text_types(parallel_pairs),
            'time_period': identify_time_period(parallel_pairs),
            'dialect_region': identify_dialect_region(parallel_pairs)
        },
        'linguistic_analysis': analyze_parallel_texts(parallel_pairs),
        'statistical_measures': calculate_statistical_measures(parallel_pairs),
        'research_applications': {
            'suggested_uses': suggest_research_applications(parallel_pairs),
            'limitations': identify_limitations(parallel_pairs)
        }
    }
    
    # Save detailed analysis in various formats
    save_analysis_files(metadata, parallel_pairs, output_folder)
    
    # Generate visualization data
    generate_visualization_data(metadata, output_folder)
    
    # Save parallel texts in multiple formats
    save_parallel_formats(parallel_pairs, output_folder)

def create_statistics_dataframe(parallel_pairs):
    """Create a DataFrame from parallel pairs statistics."""
    if not parallel_pairs:
        return pd.DataFrame()
        
    stats_data = []
    for pair in parallel_pairs:
        stats_data.append({
            'original_length': len(pair['original'].split()),
            'translation_length': len(pair['translation'].split()),
            'similarity_score': pair.get('similarity_score', 0),
            'original_text': pair['original'][:100] + '...',  # First 100 chars
            'translation_text': pair['translation'][:100] + '...'
        })
    
    return pd.DataFrame(stats_data)

def save_analysis_files(metadata, parallel_pairs, output_folder):
    """Save analysis files and create visualizations."""
    try:
        # Ensure output folder exists
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save metadata if it exists
        if metadata:
            metadata_path = output_folder / "parallel_metadata.yaml"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, allow_unicode=True)
        
        # Create and save statistical analysis if there are parallel pairs
        if parallel_pairs:
            stats_df = create_statistics_dataframe(parallel_pairs)
            if not stats_df.empty:
                stats_path = output_folder / "statistical_analysis.csv"
                stats_df.to_csv(stats_path, index=False, encoding='utf-8')
        
        # Create visualizations only if we have metadata
        if metadata and parallel_pairs:
            try:
                visualizer = ParallelTextVisualizer(output_folder)
                visualizer.create_all_visualizations(metadata)
            except Exception as viz_error:
                print(f"Warning: Visualization creation failed: {viz_error}")
                # Continue execution even if visualization fails
        
        # Save a summary report
        summary = {
            'analysis_time': datetime.datetime.now().isoformat(),
            'total_pairs': len(parallel_pairs) if parallel_pairs else 0,
            'files_created': [
                str(f.relative_to(output_folder)) 
                for f in output_folder.glob('*') 
                if f.is_file()
            ]
        }
        
        with open(output_folder / "analysis_summary.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, allow_unicode=True)
            
        return True
        
    except Exception as e:
        print(f"Error in save_analysis_files: {e}")
        import traceback
        traceback.print_exc()
        return False
def identify_phonological_patterns(parallel_pairs):
    """Identify phonological patterns in dialectal texts."""
    patterns = {
        'vowel_harmony': defaultdict(int),
        'consonant_changes': defaultdict(int),
        'sound_drops': defaultdict(int),
        'dialectal_sounds': defaultdict(int)
    }
    
    # Common Crimean Tatar dialectal patterns
    dialectal_features = {
        'vowel_changes': {
            'a': 'e',  # e.g., kalgan -> kelgen
            'ı': 'i',  # e.g., kırım -> kirim
            'o': 'u',  # e.g., yok -> yuk
            'ö': 'ü'   # e.g., köz -> küz
        },
        'consonant_changes': {
            'b': 'p',  # e.g., bar -> par
            'c': 'y',  # e.g., cok -> yok
            'ç': 'ş',  # e.g., çıktı -> şıktı
            'k': 'h'   # e.g., kalk -> halh
        }
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        
        for word in orig_words:
            # Check vowel harmony
            vowels = [c for c in word if c in 'aâeêıîiîoôöuûü']
            if vowels:
                patterns['vowel_harmony'][tuple(vowels)] += 1
            
            # Check consonant changes
            for old, new in dialectal_features['consonant_changes'].items():
                if old in word:
                    patterns['consonant_changes'][f"{old}->{new}"] += 1
            
            # Check sound drops (word-final consonant drops)
            if word.endswith(('k', 'p', 't')):
                patterns['sound_drops'][f"final_{word[-1]}"] += 1
    
    return {k: dict(v) for k, v in patterns.items()}


def find_word_context(word, text, context_window=5):
    """Find the context of a word in text."""
    words = word_tokenize(text.lower())
    contexts = []
    
    for i, w in enumerate(words):
        if w == word:
            # Get context window before and after the word
            start = max(0, i - context_window)
            end = min(len(words), i + context_window + 1)
            
            context = ' '.join(words[start:end])
            contexts.append(context)
    
    return contexts[0] if contexts else ''

def identify_dialect_terms(parallel_pairs):
    """Identify dialect-specific terms."""
    dialect_terms = defaultdict(list)
    
    # Expanded dialectal lexicon for Crimean Tatar
    dialectal_lexicon = {
        # Family and relationships
        'bala': 'çocuk',
        'emce': 'teyze',
        'kelin': 'gelin',
        'ece': 'anne',
        'aka': 'ağabey',
        
        # Time expressions
        'tünegun': 'dün',
        'bugün': 'bu gün',
        'yarın': 'yarin',
        
        # Common verbs
        'ketmek': 'gitmek',
        'kelmek': 'gelmek',
        'aytmak': 'söylemek',
        'bermek': 'vermek',
        
        # Pronouns and demonstratives
        'men': 'ben',
        'sen': 'sen',
        'o': 'o',
        'biz': 'biz',
        'siz': 'siz',
        'olar': 'onlar',
        
        # Question words
        'kayday': 'nasıl',
        'kayda': 'nerede',
        'kim': 'kim',
        'ne': 'ne',
        
        # Common nouns
        'ev': 'ev',
        'suv': 'su',
        'kün': 'gün',
        'tav': 'dağ',
        
        # Adjectives
        'yahşı': 'iyi',
        'yaman': 'kötü',
        'büyük': 'büyük',
        'kiçik': 'küçük'
    }
    
    # Regional variations dictionary
    regional_variations = {
        'northern': {
            'kelgen': 'gelen',
            'bergen': 'veren',
            'körgen': 'gören'
        },
        'central': {
            'keldi': 'geldi',
            'berdi': 'verdi',
            'kördi': 'gördü'
        },
        'southern': {
            'keldi': 'geldi',
            'berdi': 'verdi',
            'kördi': 'gördü'
        }
    }
    
    try:
        for pair in parallel_pairs:
            orig_text = pair['original'].lower()
            trans_text = pair['translation'].lower()
            
            # Tokenize texts
            orig_words = word_tokenize(orig_text)
            trans_words = word_tokenize(trans_text)
            
            # Check standard dialectal terms
            for word in orig_words:
                if word in dialectal_lexicon:
                    contexts = find_word_context(word, orig_text)
                    translation = dialectal_lexicon[word]
                    
                    dialect_terms[word].append({
                        'context': contexts,
                        'standard_form': translation,
                        'type': 'standard',
                        'frequency': orig_text.count(word)
                    })
            
            # Check regional variations
            for region, variations in regional_variations.items():
                for dialect_word, standard_word in variations.items():
                    if dialect_word in orig_words:
                        contexts = find_word_context(dialect_word, orig_text)
                        
                        dialect_terms[dialect_word].append({
                            'context': contexts,
                            'standard_form': standard_word,
                            'type': 'regional',
                            'region': region,
                            'frequency': orig_text.count(dialect_word)
                        })
        
        # Add statistical summary
        summary = {
            'total_terms_found': len(dialect_terms),
            'frequency_by_type': defaultdict(int),
            'regional_distribution': defaultdict(int)
        }
        
        for word, occurrences in dialect_terms.items():
            for occurrence in occurrences:
                summary['frequency_by_type'][occurrence['type']] += occurrence['frequency']
                if occurrence['type'] == 'regional':
                    summary['regional_distribution'][occurrence['region']] += occurrence['frequency']
        
        return {
            'terms': dict(dialect_terms),
            'summary': {
                'total_terms': summary['total_terms_found'],
                'frequency_by_type': dict(summary['frequency_by_type']),
                'regional_distribution': dict(summary['regional_distribution'])
            }
        }
        
    except Exception as e:
        print(f"Error in dialect term identification: {e}")
        return {
            'terms': {},
            'summary': {
                'total_terms': 0,
                'frequency_by_type': {},
                'regional_distribution': {}
            },
            'error': str(e)
        }
    
    
def identify_phonological_patterns(parallel_pairs):
    """Identify phonological patterns in dialectal texts."""
    patterns = {
        'vowel_harmony': defaultdict(int),
        'consonant_changes': defaultdict(int),
        'sound_drops': defaultdict(int),
        'dialectal_sounds': defaultdict(int)
    }
    
    # Common Crimean Tatar dialectal patterns
    dialectal_features = {
        'vowel_changes': {
            'a': 'e',  # e.g., kalgan -> kelgen
            'ı': 'i',  # e.g., kırım -> kirim
            'o': 'u',  # e.g., yok -> yuk
            'ö': 'ü'   # e.g., köz -> küz
        },
        'consonant_changes': {
            'b': 'p',  # e.g., bar -> par
            'c': 'y',  # e.g., cok -> yok
            'ç': 'ş',  # e.g., çıktı -> şıktı
            'k': 'h'   # e.g., kalk -> halh
        }
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        
        for word in orig_words:
            # Check vowel harmony
            vowels = [c for c in word if c in 'aâeêıîiîoôöuûü']
            if vowels:
                patterns['vowel_harmony'][tuple(vowels)] += 1
            
            # Check consonant changes
            for old, new in dialectal_features['consonant_changes'].items():
                if old in word:
                    patterns['consonant_changes'][f"{old}->{new}"] += 1
            
            # Check sound drops (word-final consonant drops)
            if word.endswith(('k', 'p', 't')):
                patterns['sound_drops'][f"final_{word[-1]}"] += 1
    
    return {k: dict(v) for k, v in patterns.items()}

def identify_morphological_patterns(parallel_pairs):
    """Identify morphological patterns specific to dialects."""
    patterns = {
        'suffixes': defaultdict(int),
        'prefix_usage': defaultdict(int),
        'tense_markers': defaultdict(int),
        'case_markers': defaultdict(int)
    }
    
    # Crimean Tatar specific morphological features
    dialectal_morphology = {
        'suffixes': ['lar', 'ler', 'day', 'dey', 'çe', 'ca'],
        'tense_markers': ['yor', 'acak', 'ecek', 'gan', 'gen'],
        'case_markers': ['nıñ', 'niñ', 'ga', 'ge', 'nı', 'ni']
    }
    
    for pair in parallel_pairs:
        words = word_tokenize(pair['original'].lower())
        
        for word in words:
            # Check suffixes
            for suffix in dialectal_morphology['suffixes']:
                if word.endswith(suffix):
                    patterns['suffixes'][suffix] += 1
            
            # Check tense markers
            for marker in dialectal_morphology['tense_markers']:
                if marker in word:
                    patterns['tense_markers'][marker] += 1
            
            # Check case markers
            for marker in dialectal_morphology['case_markers']:
                if word.endswith(marker):
                    patterns['case_markers'][marker] += 1
    
    return {k: dict(v) for k, v in patterns.items()}

def identify_dialect_terms(parallel_pairs):
    """Identify dialect-specific terms."""
    dialect_terms = defaultdict(list)
    
    # Known dialectal terms (Crimean Tatar examples)
    dialectal_lexicon = {
        'bala': 'çocuk',
        'tünegun': 'dün',
        'emce': 'teyze',
        'kelin': 'gelin'
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        trans_words = word_tokenize(pair['translation'].lower())
        
        for word in orig_words:
            if word in dialectal_lexicon:
                context = find_word_context(word, pair['original'])
                translation = dialectal_lexicon[word]
                dialect_terms[word].append({
                    'context': context,
                    'standard_form': translation
                })
    
    return dict(dialect_terms)

def analyze_regional_variations(parallel_pairs):
    """Analyze regional variations in language use."""
    variations = {
        'northern': defaultdict(int),
        'central': defaultdict(int),
        'southern': defaultdict(int)
    }
    
    # Regional variation markers
    regional_features = {
        'northern': ['ga', 'ge', 'ka', 'ke'],
        'central': ['a', 'e', 'da', 'de'],
        'southern': ['ğa', 'ğe', 'ña', 'ñe']
    }
    
    for pair in parallel_pairs:
        text = pair['original'].lower()
        
        for region, features in regional_features.items():
            for feature in features:
                count = text.count(feature)
                if count > 0:
                    variations[region][feature] = count
    
    return {region: dict(counts) for region, counts in variations.items()}
def analyze_rhyme_patterns(parallel_pairs):
    """Analyze rhyme patterns in parallel texts."""
    def get_rhyme_end(word):
        """Get the rhyming part of a word."""
        vowels = 'aâeêıîiîoôöuûü'
        word = word.lower()
        # Find last vowel position
        for i in range(len(word)-1, -1, -1):
            if word[i] in vowels:
                return word[i:]
        return word[-2:] if len(word) > 2 else word

    def find_rhyme_scheme(lines):
        """Identify rhyme scheme in a group of lines."""
        if not lines:
            return ''
        
        rhyme_ends = [get_rhyme_end(line.split()[-1]) for line in lines if line.strip()]
        scheme = ''
        rhyme_map = {}
        current_letter = 'a'
        
        for end in rhyme_ends:
            if end not in rhyme_map:
                rhyme_map[end] = current_letter
                current_letter = chr(ord(current_letter) + 1)
            scheme += rhyme_map[end]
        
        return scheme

    rhyme_analysis = {
        'patterns': defaultdict(int),
        'preserved_rhymes': 0,
        'total_rhyming_lines': 0,
        'rhyme_schemes': defaultdict(int)
    }

    try:
        for pair in parallel_pairs:
            orig_lines = pair['original'].split('\n')
            trans_lines = pair['translation'].split('\n')
            
            # Analyze original text rhyme scheme
            orig_scheme = find_rhyme_scheme(orig_lines)
            if orig_scheme:
                rhyme_analysis['rhyme_schemes'][orig_scheme] += 1
            
            # Compare rhyming in original and translation
            for orig_line, trans_line in zip(orig_lines, trans_lines):
                if not (orig_line.strip() and trans_line.strip()):
                    continue
                    
                orig_rhyme = get_rhyme_end(orig_line.split()[-1])
                trans_rhyme = get_rhyme_end(trans_line.split()[-1])
                
                if orig_rhyme:
                    rhyme_analysis['total_rhyming_lines'] += 1
                    if trans_rhyme and len(trans_rhyme) >= 2:
                        rhyme_analysis['preserved_rhymes'] += 1
                        rhyme_analysis['patterns'][f"{orig_rhyme}->{trans_rhyme}"] += 1
        
        # Calculate preservation ratio
        if rhyme_analysis['total_rhyming_lines'] > 0:
            rhyme_analysis['preservation_ratio'] = (
                rhyme_analysis['preserved_rhymes'] / rhyme_analysis['total_rhyming_lines']
            )
        else:
            rhyme_analysis['preservation_ratio'] = 0.0
            
        return dict(rhyme_analysis)
        
    except Exception as e:
        print(f"Error in rhyme analysis: {e}")
        return {
            'patterns': {},
            'preserved_rhymes': 0,
            'total_rhyming_lines': 0,
            'rhyme_schemes': {},
            'preservation_ratio': 0.0,
            'error': str(e)
        }

def identify_alliteration(parallel_pairs):
    """Identify alliteration patterns in texts."""
    def find_alliteration(text):
        """Find alliteration in a single text."""
        words = text.lower().split()
        patterns = defaultdict(int)
        
        for i in range(len(words) - 1):
            if len(words[i]) > 0 and len(words[i+1]) > 0:
                if words[i][0] == words[i+1][0]:
                    pattern = words[i][0]
                    patterns[pattern] += 1
        
        return dict(patterns)

    alliteration_analysis = {
        'original_patterns': defaultdict(int),
        'translation_patterns': defaultdict(int),
        'preserved_patterns': 0,
        'total_patterns': 0
    }

    try:
        for pair in parallel_pairs:
            orig_patterns = find_alliteration(pair['original'])
            trans_patterns = find_alliteration(pair['translation'])
            
            # Update pattern counts
            for pattern, count in orig_patterns.items():
                alliteration_analysis['original_patterns'][pattern] += count
                alliteration_analysis['total_patterns'] += count
                
                # Check if pattern is preserved in translation
                if pattern in trans_patterns:
                    alliteration_analysis['preserved_patterns'] += min(count, trans_patterns[pattern])
        
        # Calculate preservation ratio
        if alliteration_analysis['total_patterns'] > 0:
            alliteration_analysis['preservation_ratio'] = (
                alliteration_analysis['preserved_patterns'] / alliteration_analysis['total_patterns']
            )
        else:
            alliteration_analysis['preservation_ratio'] = 0.0
            
        return dict(alliteration_analysis)
        
    except Exception as e:
        print(f"Error in alliteration analysis: {e}")
        return {
            'original_patterns': {},
            'translation_patterns': {},
            'preserved_patterns': 0,
            'total_patterns': 0,
            'preservation_ratio': 0.0,
            'error': str(e)
        }


def identify_metaphors(parallel_pairs):
    """Identify metaphors and their translations."""
    # Common metaphor markers in Ottoman Turkish
    metaphor_markers = {
        'comparison': ['gibi', 'kadar', 'sanki', 'benzer'],
        'possessive': ['gözü', 'kalbi', 'ruhu', 'dili'],
        'nature': ['gül', 'bülbül', 'ay', 'güneş', 'deniz'],
        'abstract': ['aşk', 'gönül', 'ruh', 'can', 'hayat']
    }

    metaphor_analysis = {
        'identified_metaphors': [],
        'preservation_stats': defaultdict(int),
        'metaphor_types': defaultdict(int)
    }

    try:
        for pair in parallel_pairs:
            orig_text = pair['original'].lower()
            trans_text = pair['translation'].lower()
            
            # Search for metaphors using markers
            for category, markers in metaphor_markers.items():
                for marker in markers:
                    if marker in orig_text:
                        # Find context around metaphor
                        context = find_word_context(marker, orig_text)
                        trans_context = find_word_context(marker, trans_text)
                        
                        metaphor_analysis['metaphor_types'][category] += 1
                        
                        metaphor_info = {
                            'type': category,
                            'marker': marker,
                            'original_context': context,
                            'translation_context': trans_context,
                            'preserved': bool(trans_context)
                        }
                        
                        metaphor_analysis['identified_metaphors'].append(metaphor_info)
                        metaphor_analysis['preservation_stats']['total'] += 1
                        if metaphor_info['preserved']:
                            metaphor_analysis['preservation_stats']['preserved'] += 1
        
        # Calculate preservation ratio
        if metaphor_analysis['preservation_stats']['total'] > 0:
            metaphor_analysis['preservation_stats']['ratio'] = (
                metaphor_analysis['preservation_stats']['preserved'] /
                metaphor_analysis['preservation_stats']['total']
            )
        else:
            metaphor_analysis['preservation_stats']['ratio'] = 0.0
            
        return dict(metaphor_analysis)
        
    except Exception as e:
        print(f"Error in metaphor analysis: {e}")
        return {
            'identified_metaphors': [],
            'preservation_stats': {'total': 0, 'preserved': 0, 'ratio': 0.0},
            'metaphor_types': {},
            'error': str(e)
        }

def analyze_poetic_features(parallel_pairs):
    """Analyze poetic features in texts."""
    try:
        features = {
            'meter': analyze_meter_preservation(parallel_pairs),
            'rhyme': analyze_rhyme_patterns(parallel_pairs),
            'alliteration': identify_alliteration(parallel_pairs),
            'metaphors': identify_metaphors(parallel_pairs),
            'summary': {
                'poetic_density': 0.0,
                'preservation_score': 0.0,
                'feature_counts': defaultdict(int)
            }
        }
        
        # Calculate overall poetic density and preservation
        total_features = sum(1 for f in [features['meter'], features['rhyme'],
                                       features['alliteration'], features['metaphors']]
                           if f and not isinstance(f, str))
        
        if total_features > 0:
            preservation_scores = []
            if 'preserved_ratio' in features['meter']:
                preservation_scores.append(features['meter']['preserved_ratio'])
            if 'preservation_ratio' in features['rhyme']:
                preservation_scores.append(features['rhyme']['preservation_ratio'])
            if 'preservation_ratio' in features['alliteration']:
                preservation_scores.append(features['alliteration']['preservation_ratio'])
            if 'preservation_stats' in features['metaphors']:
                preservation_scores.append(features['metaphors']['preservation_stats']['ratio'])
            
            features['summary']['preservation_score'] = (
                sum(preservation_scores) / len(preservation_scores)
                if preservation_scores else 0.0
            )
            
            features['summary']['poetic_density'] = total_features / len(parallel_pairs)
        
        return features
        
    except Exception as e:
        print(f"Error in poetic feature analysis: {e}")
        return {
            'meter': {},
            'rhyme': {},
            'alliteration': {},
            'metaphors': {},
            'summary': {
                'poetic_density': 0.0,
                'preservation_score': 0.0,
                'feature_counts': {}
            },
            'error': str(e)
        }
    
def analyze_poetic_features(parallel_pairs):
    """Analyze poetic features in texts."""
    features = {
        'meter': analyze_meter_preservation(parallel_pairs),
        'rhyme': analyze_rhyme_patterns(parallel_pairs),
        'alliteration': identify_alliteration(parallel_pairs),
        'metaphors': identify_metaphors(parallel_pairs)
    }
    return features

def assess_formality(parallel_pairs):
    """Assess the level of formality in texts."""
    formality_markers = {
        'formal': ['efendi', 'hazret', 'cenap', 'zatıaliniz', 'buyurmak'],
        'informal': ['be', 'ulan', 'yahu', 'hey', 'abe'],
        'neutral': ['demek', 'söylemek', 'gelmek', 'gitmek']
    }
    
    scores = []
    for pair in parallel_pairs:
        text = pair['original'].lower()
        formal_count = sum(text.count(marker) for marker in formality_markers['formal'])
        informal_count = sum(text.count(marker) for marker in formality_markers['informal'])
        
        total = formal_count + informal_count
        if total > 0:
            formality_score = formal_count / total
            scores.append(formality_score)
    
    return {
        'average_formality': np.mean(scores) if scores else 0.5,
        'formality_range': (min(scores), max(scores)) if scores else (0.5, 0.5),
        'formal_markers_found': formal_count,
        'informal_markers_found': informal_count
    }

def identify_register_shifts(parallel_pairs):
    """Identify shifts in register between original and translation."""
    shifts = []
    
    for pair in parallel_pairs:
        orig_formality = assess_formality([{'original': pair['original']}])
        trans_formality = assess_formality([{'original': pair['translation']}])
        
        if abs(orig_formality['average_formality'] - trans_formality['average_formality']) > 0.2:
            shifts.append({
                'original_text': pair['original'][:100],
                'translation_text': pair['translation'][:100],
                'formality_change': trans_formality['average_formality'] - orig_formality['average_formality']
            })
    
    return {
        'total_shifts': len(shifts),
        'significant_shifts': shifts[:5],  # Return top 5 most significant shifts
        'average_shift': np.mean([s['formality_change'] for s in shifts]) if shifts else 0.0
    }

def identify_rhetorical_devices(parallel_pairs):
    """Identify rhetorical devices in texts."""
    devices = {
        'metaphors': defaultdict(int),
        'similes': defaultdict(int),
        'personification': defaultdict(int),
        'repetition': defaultdict(int)
    }
    
    # Common markers for rhetorical devices
    markers = {
        'similes': ['gibi', 'kadar', 'sanki', 'benzer'],
        'personification': ['söyledi', 'ağladı', 'güldü', 'düşündü'],
        'repetition_threshold': 3
    }
    
    for pair in parallel_pairs:
        text = pair['original'].lower()
        words = word_tokenize(text)
        
        # Check for similes
        for marker in markers['similes']:
            count = text.count(marker)
            if count > 0:
                devices['similes'][marker] += count
        
        # Check for personification
        for marker in markers['personification']:
            if marker in text:
                devices['personification'][marker] += 1
        
        # Check for repetition
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if count >= markers['repetition_threshold']:
                devices['repetition'][word] = count
    
    return {k: dict(v) for k, v in devices.items()}


def identify_morphological_patterns(parallel_pairs):
    """Identify morphological patterns specific to dialects."""
    patterns = {
        'suffixes': defaultdict(int),
        'prefix_usage': defaultdict(int),
        'tense_markers': defaultdict(int),
        'case_markers': defaultdict(int)
    }
    
    # Crimean Tatar specific morphological features
    dialectal_morphology = {
        'suffixes': ['lar', 'ler', 'day', 'dey', 'çe', 'ca'],
        'tense_markers': ['yor', 'acak', 'ecek', 'gan', 'gen'],
        'case_markers': ['nıñ', 'niñ', 'ga', 'ge', 'nı', 'ni']
    }
    
    for pair in parallel_pairs:
        words = word_tokenize(pair['original'].lower())
        
        for word in words:
            # Check suffixes
            for suffix in dialectal_morphology['suffixes']:
                if word.endswith(suffix):
                    patterns['suffixes'][suffix] += 1
            
            # Check tense markers
            for marker in dialectal_morphology['tense_markers']:
                if marker in word:
                    patterns['tense_markers'][marker] += 1
            
            # Check case markers
            for marker in dialectal_morphology['case_markers']:
                if word.endswith(marker):
                    patterns['case_markers'][marker] += 1
    
    return {k: dict(v) for k, v in patterns.items()}

def identify_dialect_terms(parallel_pairs):
    """Identify dialect-specific terms."""
    dialect_terms = defaultdict(list)
    
    # Known dialectal terms (Crimean Tatar examples)
    dialectal_lexicon = {
        'bala': 'çocuk',
        'tünegun': 'dün',
        'emce': 'teyze',
        'kelin': 'gelin'
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        trans_words = word_tokenize(pair['translation'].lower())
        
        for word in orig_words:
            if word in dialectal_lexicon:
                context = find_word_context(word, pair['original'])
                translation = dialectal_lexicon[word]
                dialect_terms[word].append({
                    'context': context,
                    'standard_form': translation
                })
    
    return dict(dialect_terms)

def analyze_regional_variations(parallel_pairs):
    """Analyze regional variations in language use."""
    variations = {
        'northern': defaultdict(int),
        'central': defaultdict(int),
        'southern': defaultdict(int)
    }
    
    # Regional variation markers
    regional_features = {
        'northern': ['ga', 'ge', 'ka', 'ke'],
        'central': ['a', 'e', 'da', 'de'],
        'southern': ['ğa', 'ğe', 'ña', 'ñe']
    }
    
    for pair in parallel_pairs:
        text = pair['original'].lower()
        
        for region, features in regional_features.items():
            for feature in features:
                count = text.count(feature)
                if count > 0:
                    variations[region][feature] = count
    
    return {region: dict(counts) for region, counts in variations.items()}

def analyze_poetic_features(parallel_pairs):
    """Analyze poetic features in texts."""
    features = {
        'meter': analyze_meter_preservation(parallel_pairs),
        'rhyme': analyze_rhyme_patterns(parallel_pairs),
        'alliteration': identify_alliteration(parallel_pairs),
        'metaphors': identify_metaphors(parallel_pairs)
    }
    return features

def assess_formality(parallel_pairs):
    """Assess the level of formality in texts."""
    formality_markers = {
        'formal': ['efendi', 'hazret', 'cenap', 'zatıaliniz', 'buyurmak'],
        'informal': ['be', 'ulan', 'yahu', 'hey', 'abe'],
        'neutral': ['demek', 'söylemek', 'gelmek', 'gitmek']
    }
    
    scores = []
    for pair in parallel_pairs:
        text = pair['original'].lower()
        formal_count = sum(text.count(marker) for marker in formality_markers['formal'])
        informal_count = sum(text.count(marker) for marker in formality_markers['informal'])
        
        total = formal_count + informal_count
        if total > 0:
            formality_score = formal_count / total
            scores.append(formality_score)
    
    return {
        'average_formality': np.mean(scores) if scores else 0.5,
        'formality_range': (min(scores), max(scores)) if scores else (0.5, 0.5),
        'formal_markers_found': formal_count,
        'informal_markers_found': informal_count
    }

def identify_register_shifts(parallel_pairs):
    """Identify shifts in register between original and translation."""
    shifts = []
    
    for pair in parallel_pairs:
        orig_formality = assess_formality([{'original': pair['original']}])
        trans_formality = assess_formality([{'original': pair['translation']}])
        
        if abs(orig_formality['average_formality'] - trans_formality['average_formality']) > 0.2:
            shifts.append({
                'original_text': pair['original'][:100],
                'translation_text': pair['translation'][:100],
                'formality_change': trans_formality['average_formality'] - orig_formality['average_formality']
            })
    
    return {
        'total_shifts': len(shifts),
        'significant_shifts': shifts[:5],  # Return top 5 most significant shifts
        'average_shift': np.mean([s['formality_change'] for s in shifts]) if shifts else 0.0
    }

def identify_rhetorical_devices(parallel_pairs):
    """Identify rhetorical devices in texts."""
    devices = {
        'metaphors': defaultdict(int),
        'similes': defaultdict(int),
        'personification': defaultdict(int),
        'repetition': defaultdict(int)
    }
    
    # Common markers for rhetorical devices
    markers = {
        'similes': ['gibi', 'kadar', 'sanki', 'benzer'],
        'personification': ['söyledi', 'ağladı', 'güldü', 'düşündü'],
        'repetition_threshold': 3
    }
    
    for pair in parallel_pairs:
        text = pair['original'].lower()
        words = word_tokenize(text)
        
        # Check for similes
        for marker in markers['similes']:
            count = text.count(marker)
            if count > 0:
                devices['similes'][marker] += count
        
        # Check for personification
        for marker in markers['personification']:
            if marker in text:
                devices['personification'][marker] += 1
        
        # Check for repetition
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if count >= markers['repetition_threshold']:
                devices['repetition'][word] = count
    
    return {k: dict(v) for k, v in devices.items()}
    
def generate_visualization_data(metadata, output_folder):
    """Generate data for visualizations."""
    viz_data = {
        'diachronic_changes': prepare_diachronic_visualization(metadata),
        'dialectal_features': prepare_dialectal_visualization(metadata),
        'statistical_measures': prepare_statistical_visualization(metadata)
    }
    
    with open(output_folder / "visualization_data.json", 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)

def analyze_parallel_texts(parallel_pairs):
    """Comprehensive analysis of parallel texts."""
    from collections import Counter
    import numpy as np
    
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
        },
        'diachronic_analysis': analyze_diachronic_changes(parallel_pairs),
        'dialectal_features': analyze_dialectal_features(parallel_pairs),
        'stylistic_analysis': analyze_style(parallel_pairs)
    }
    return analysis

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
    
def calculate_lexical_retention(parallel_pairs):
    """Calculate the rate of lexical retention between versions."""
    ottoman_lexicon = {
        'archaic': set(['müşkül', 'mahzun', 'mektup', 'kitap', 'kalem']),
        'religious': set(['bismillah', 'elhamdülillah', 'inşallah']),
        'administrative': set(['ferman', 'irade', 'tezkere']),
        'cultural': set(['efendi', 'ağa', 'hanım', 'bey'])
    }
    
    retention_stats = {
        'total_words': 0,
        'retained_words': 0,
        'retention_by_category': defaultdict(float)
    }
    
    for pair in parallel_pairs:
        orig_words = set(word_tokenize(pair['original'].lower()))
        trans_words = set(word_tokenize(pair['translation'].lower()))
        
        retention_stats['total_words'] += len(orig_words)
        retention_stats['retained_words'] += len(orig_words & trans_words)
        
        for category, word_set in ottoman_lexicon.items():
            orig_category_words = orig_words & word_set
            retained_category_words = trans_words & word_set
            if orig_category_words:
                retention_stats['retention_by_category'][category] = (
                    len(retained_category_words) / len(orig_category_words)
                )
    
    return {
        'overall_retention': retention_stats['retained_words'] / retention_stats['total_words']
        if retention_stats['total_words'] > 0 else 0,
        'by_category': dict(retention_stats['retention_by_category'])
    }

def identify_modernization_patterns(parallel_pairs):
    """Identify patterns of lexical modernization."""
    modernization_patterns = {
        'suffix_changes': defaultdict(int),
        'root_changes': defaultdict(int),
        'loan_word_replacements': defaultdict(int)
    }
    
    common_modernizations = {
        'suffixes': {
            'iyet': 'lik',
            'at': 'lar',
            'iyat': 'ler'
        },
        'roots': {
            'mektep': 'okul',
            'muallim': 'öğretmen',
            'talebe': 'öğrenci'
        }
    }
    
    for pair in parallel_pairs:
        orig_text = pair['original'].lower()
        trans_text = pair['translation'].lower()
        
        # Check suffix changes
        for old_suffix, new_suffix in common_modernizations['suffixes'].items():
            if old_suffix in orig_text and new_suffix in trans_text:
                modernization_patterns['suffix_changes'][f"{old_suffix}->{new_suffix}"] += 1
        
        # Check root changes
        for old_root, new_root in common_modernizations['roots'].items():
            if old_root in orig_text and new_root in trans_text:
                modernization_patterns['root_changes'][f"{old_root}->{new_root}"] += 1
    
    return dict(modernization_patterns)
def analyze_sentence_structure(sentence):
    """Analyze the basic word order of a sentence."""
    try:
        words = word_tokenize(sentence.lower())
        
        # Basic POS patterns for Turkish
        verb_markers = {'mek', 'mak', 'di', 'dı', 'du', 'dü', 'ti', 'tı', 'tu', 'tü', 'yor', 'ecek', 'acak'}
        subject_markers = {'ben', 'sen', 'o', 'biz', 'siz', 'onlar'}
        object_markers = {'i', 'ı', 'u', 'ü', 'yi', 'yı', 'yu', 'yü'}
        
        # Find components
        verb_position = -1
        subject_position = -1
        object_position = -1
        
        for i, word in enumerate(words):
            # Check for verb
            if any(word.endswith(marker) for marker in verb_markers):
                verb_position = i
                break
            
            # Check for subject
            if word in subject_markers:
                subject_position = i
            
            # Check for object (simple check for accusative case)
            if any(word.endswith(marker) for marker in object_markers):
                object_position = i
        
        # Determine word order
        if -1 not in (subject_position, object_position, verb_position):
            positions = [
                ('S', subject_position),
                ('O', object_position),
                ('V', verb_position)
            ]
            positions.sort(key=lambda x: x[1])
            return ''.join(pos[0] for pos in positions)
        
        return 'SOV'  # Default for Turkish
        
    except Exception as e:
        print(f"Error in sentence structure analysis: {e}")
        return 'UNKNOWN'


def calculate_complexity_measures(text):
    """Calculate various syntactic complexity measures for a text."""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Analyze clauses
        clause_analysis = analyze_clauses(text)
        clauses = clause_analysis['clauses']
        
        # Calculate measures
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate clause density (clauses per sentence)
        clause_density = len(clauses) / len(sentences) if sentences else 0
        
        # Calculate embedding depth
        embedding_depth = calculate_embedding_depth(clauses)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'clause_density': clause_density,
            'embedding_depth': embedding_depth
        }
        
    except Exception as e:
        print(f"Error in complexity measurement: {e}")
        return {
            'avg_sentence_length': 0,
            'clause_density': 0,
            'embedding_depth': 0
        }

def calculate_embedding_depth(clauses):
    """Calculate the maximum embedding depth of clauses."""
    try:
        max_depth = 0
        current_depth = 0
        
        for clause in clauses:
            if clause['type'] == 'subordinate':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif clause['type'] == 'main':
                current_depth = 0
        
        return max_depth
        
    except Exception as e:
        print(f"Error in embedding depth calculation: {e}")
        return 0

def compare_syntactic_complexity(parallel_pairs):
    """Compare syntactic complexity between versions."""
    complexity_measures = {
        'original': {
            'avg_sentence_length': 0,
            'clause_density': 0,
            'embedding_depth': 0
        },
        'translation': {
            'avg_sentence_length': 0,
            'clause_density': 0,
            'embedding_depth': 0
        }
    }
    
    for pair in parallel_pairs:
        # Calculate complexity measures for both versions
        orig_measures = calculate_complexity_measures(pair['original'])
        trans_measures = calculate_complexity_measures(pair['translation'])
        
        for measure in ['avg_sentence_length', 'clause_density', 'embedding_depth']:
            complexity_measures['original'][measure] += orig_measures[measure]
            complexity_measures['translation'][measure] += trans_measures[measure]
    
    # Average the measures
    pair_count = len(parallel_pairs)
    if pair_count > 0:
        for version in ['original', 'translation']:
            for measure in complexity_measures[version]:
                complexity_measures[version][measure] /= pair_count
    
    return complexity_measures

def analyze_clauses(text):
    """Analyze clause structure in text."""
    try:
        clauses = []
        sentences = sent_tokenize(text)
        
        # Clause markers in Turkish
        subordinate_markers = {
            'ki', 'dığı', 'diği', 'duğu', 'düğü',
            'ince', 'arak', 'erek', 'ip', 'up',
            'meden', 'madan', 'ken'
        }
        
        coordinate_markers = {
            've', 'veya', 'ama', 'fakat', 'çünkü',
            'ancak', 'lakin', 'ya da'
        }
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            current_clause = []
            
            for i, word in enumerate(words):
                current_clause.append(word)
                
                # Check for clause boundaries
                if any(word.endswith(marker) for marker in subordinate_markers):
                    clauses.append({
                        'type': 'subordinate',
                        'text': ' '.join(current_clause),
                        'marker': word
                    })
                    current_clause = []
                
                elif word in coordinate_markers:
                    clauses.append({
                        'type': 'coordinate',
                        'text': ' '.join(current_clause[:-1]),  # Exclude the coordinator
                        'marker': word
                    })
                    current_clause = []
            
            if current_clause:
                clauses.append({
                    'type': 'main',
                    'text': ' '.join(current_clause),
                    'marker': None
                })
        
        return {
            'clauses': clauses,
            'statistics': {
                'total_clauses': len(clauses),
                'subordinate': sum(1 for c in clauses if c['type'] == 'subordinate'),
                'coordinate': sum(1 for c in clauses if c['type'] == 'coordinate'),
                'main': sum(1 for c in clauses if c['type'] == 'main')
            }
        }
        
    except Exception as e:
        print(f"Error in clause analysis: {e}")
        return {
            'clauses': [],
            'statistics': {
                'total_clauses': 0,
                'subordinate': 0,
                'coordinate': 0,
                'main': 0
            }
        }

def find_word_context(word, text, window_size=5):
    """Find the context of a word in text with specified window size."""
    try:
        words = word_tokenize(text.lower())
        contexts = []
        
        for i, w in enumerate(words):
            if w == word.lower():
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                context = ' '.join(words[start:end])
                contexts.append({
                    'text': context,
                    'position': i,
                    'preceding': ' '.join(words[start:i]),
                    'following': ' '.join(words[i+1:end])
                })
        
        return contexts[0] if contexts else {'text': '', 'position': -1, 'preceding': '', 'following': ''}
        
    except Exception as e:
        print(f"Error in context finding: {e}")
        return {'text': '', 'position': -1, 'preceding': '', 'following': ''}
    
def identify_archaic_terms(parallel_pairs):
    from collections import defaultdict
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    """Identify archaic terms and their modern equivalents."""
    archaic_terms = defaultdict(list)
    
    archaic_dictionary = {
        'eylemek': 'yapmak',
        'kılmak': 'yapmak',
        'buyurmak': 'söylemek',
        'ziyade': 'çok',
        'pek': 'çok'
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        
        for word in orig_words:
            if word in archaic_dictionary:
                context = find_word_context(word, pair['original'])
                modern_form = archaic_dictionary[word]
                
                archaic_terms[word].append({
                    'modern_form': modern_form,
                    'context': context,
                    'frequency': pair['original'].lower().count(word)
                })
    
    return dict(archaic_terms)

def analyze_word_order_changes(parallel_pairs):
    """Analyze changes in word order between versions."""
    order_patterns = {
        'sov_to_svo': 0,
        'preserved_order': 0,
        'total_sentences': 0
    }
    
    for pair in parallel_pairs:
        orig_sentences = sent_tokenize(pair['original'])
        trans_sentences = sent_tokenize(pair['translation'])
        
        order_patterns['total_sentences'] += len(orig_sentences)
        
        for orig_sent, trans_sent in zip(orig_sentences, trans_sentences):
            orig_order = analyze_sentence_structure(orig_sent)
            trans_order = analyze_sentence_structure(trans_sent)
            
            if orig_order == trans_order:
                order_patterns['preserved_order'] += 1
            elif orig_order == 'SOV' and trans_order == 'SVO':
                order_patterns['sov_to_svo'] += 1
    
    return order_patterns

def analyze_clause_structure(parallel_pairs):
    """Analyze changes in clause structure."""
    structure_changes = {
        'subordination_patterns': defaultdict(int),
        'coordination_patterns': defaultdict(int),
        'complexity_shifts': []
    }
    
    for pair in parallel_pairs:
        orig_clauses = analyze_clauses(pair['original'])
        trans_clauses = analyze_clauses(pair['translation'])
        
        # Compare clause structures
        structure_changes['complexity_shifts'].append({
            'original_complexity': len(orig_clauses),
            'translation_complexity': len(trans_clauses)
        })
    
    return structure_changes

def compare_syntactic_complexity(parallel_pairs):
    """Compare syntactic complexity between versions."""
    complexity_measures = {
        'original': {
            'avg_sentence_length': 0,
            'clause_density': 0,
            'embedding_depth': 0
        },
        'translation': {
            'avg_sentence_length': 0,
            'clause_density': 0,
            'embedding_depth': 0
        }
    }
    
    for pair in parallel_pairs:
        # Calculate complexity measures for both versions
        orig_measures = calculate_complexity_measures(pair['original'])
        trans_measures = calculate_complexity_measures(pair['translation'])
        
        for measure in ['avg_sentence_length', 'clause_density', 'embedding_depth']:
            complexity_measures['original'][measure] += orig_measures[measure]
            complexity_measures['translation'][measure] += trans_measures[measure]
    
    # Average the measures
    pair_count = len(parallel_pairs)
    if pair_count > 0:
        for version in ['original', 'translation']:
            for measure in complexity_measures[version]:
                complexity_measures[version][measure] /= pair_count
    
    return complexity_measures

def identify_semantic_changes(original_text, translation_text):
    """Identify semantic changes between original and translated texts."""
    try:
        # Initialize semantic analysis
        semantic_analysis = {
            'preservation_score': 0.0,
            'field_changes': defaultdict(int),
            'semantic_shifts': [],
            'preserved_concepts': [],
            'modified_concepts': []
        }
        
        # Semantic field categories
        semantic_fields = { 
            'religious': [
                'allah', 'bismillah', 'dua', 'namaz', 'ibadet', 
                'cami', 'mümin', 'müslüman', 'din', 'iman',
                'secde', 'abdest', 'ezan', 'kuran', 'hadis',
                'cennet', 'cehennem', 'sevap', 'günah', 'helal',
                'haram', 'mübarek', 'rahmet', 'bereket', 'şükür'
            ],
    
            'cultural': [
                'adet', 'töre', 'gelenek', 'bayram', 'düğün',
                'misafir', 'sofra', 'çeyiz', 'nişan', 'kına',
                'sünnet', 'mevlit', 'helva', 'lokma', 'pilav',
                'hamam', 'kahve', 'çarşı', 'pazar', 'bedesten',
                'han', 'kervan', 'esnaf', 'lonca', 'çırak'
            ],
    
            'emotional': [
                'aşk', 'sevda', 'hüzün', 'hasret', 'gam',
                'keder', 'neşe', 'sevinç', 'mutluluk', 'elem',
                'dert', 'acı', 'ızdırap', 'özlem', 'vuslat',
                'firkat', 'hicran', 'saadet', 'meserret', 'şevk',
                'melal', 'yeis', 'ümit', 'arzu', 'heves'
            ],
    
            'nature': [
                'gül', 'bülbül', 'bahçe', 'deniz', 'dağ',
                'çiçek', 'ağaç', 'yaprak', 'çimen', 'orman',
                'nehir', 'ırmak', 'göl', 'pınar', 'çeşme',
                'güneş', 'ay', 'yıldız', 'gök', 'bulut',
                'yağmur', 'kar', 'rüzgar', 'toprak', 'taş'
            ],
            
            'abstract': [
                'zaman', 'hayat', 'ömür', 'kader', 'felek',
                'devran', 'ecel', 'baht', 'talih', 'nasip',
                'kismet', 'rüya', 'hayal', 'düş', 'gerçek',
                'varlık', 'yokluk', 'mana', 'hikmet', 'hakikat',
                'dünya', 'ahiret', 'ruh', 'can', 'nefes'
            ],
            
            'social_relations': [
                'dost', 'arkadaş', 'yaren', 'ahbap', 'yoldaş',
                'kardeş', 'bacı', 'ağabey', 'abla', 'akraba',
                'komşu', 'hısım', 'akran', 'eş', 'yar',
                'sevgili', 'canan', 'dilber', 'güzel', 'mahbub',
                'padişah', 'vezir', 'ağa', 'efendi', 'hanım'
            ],
            
            'intellectual': [
                'ilim', 'irfan', 'marifet', 'hikmet', 'fehim',
                'akıl', 'fikir', 'zihin', 'idrak', 'izan',
                'mantık', 'muhakeme', 'tefekkür', 'tasavvur', 'teemmül',
                'kitap', 'kalem', 'medrese', 'mektep', 'müderris',
                'talebe', 'üstat', 'alim', 'arif', 'hakim'
            ],
            
            'moral_values': [
                'edep', 'haya', 'iffet', 'ismet', 'fazilet',
                'ahlak', 'terbiye', 'namus', 'şeref', 'haysiyet',
                'merhamet', 'şefkat', 'vefa', 'sadakat', 'doğruluk',
                'adalet', 'insaf', 'vicdan', 'mürüvvet', 'kerem',
                'cömertlik', 'ihsan', 'lutuf', 'kerem', 'himmet'
            ],
            
            'material_culture': [
                'ev', 'konak', 'saray', 'köşk', 'yalı',
                'bahçe', 'avlu', 'sofa', 'oda', 'divanhane',
                'kilim', 'halı', 'seccade', 'mindel', 'yastık',
                'mangal', 'tencere', 'ibrik', 'tas', 'fincan',
                'kılıç', 'kalkan', 'ok', 'yay', 'mızrak'
            ],
            
            'time_concepts': [
                'sabah', 'öğle', 'akşam', 'gece', 'seher',
                'bahar', 'yaz', 'güz', 'kış', 'mevsim',
                'gün', 'ay', 'yıl', 'asır', 'devir',
                'dem', 'an', 'lahza', 'vakit', 'zaman',
                'dün', 'bugün', 'yarın', 'evvel', 'ahir'
                    ]
        }
        
        # Tokenize texts
        orig_words = word_tokenize(original_text.lower())
        trans_words = word_tokenize(translation_text.lower())
        
        # Analyze semantic fields
        for field, terms in semantic_fields.items():
            orig_field_count = sum(1 for word in orig_words if word in terms)
            trans_field_count = sum(1 for word in trans_words if word in terms)
            
            if orig_field_count > 0:
                semantic_analysis['field_changes'][field] = (
                    trans_field_count / orig_field_count
                    if orig_field_count > 0 else 0.0
                )
        
        # Analyze concept preservation
        for field, terms in semantic_fields.items():
            for term in terms:
                if term in orig_words:
                    context = find_word_context(term, original_text)
                    trans_context = find_word_context(term, translation_text)
                    
                    if trans_context['text']:
                        semantic_analysis['preserved_concepts'].append({
                            'term': term,
                            'field': field,
                            'original_context': context,
                            'translation_context': trans_context
                        })
                    else:
                        semantic_analysis['modified_concepts'].append({
                            'term': term,
                            'field': field,
                            'original_context': context
                        })
        
        # Calculate preservation score
        total_concepts = len(semantic_analysis['preserved_concepts']) + len(semantic_analysis['modified_concepts'])
        if total_concepts > 0:
            semantic_analysis['preservation_score'] = (
                len(semantic_analysis['preserved_concepts']) / total_concepts
            )
        
        return semantic_analysis
        
    except Exception as e:
        print(f"Error in semantic change identification: {e}")
        return {
            'preservation_score': 0.0,
            'field_changes': {},
            'semantic_shifts': [],
            'preserved_concepts': [],
            'modified_concepts': [],
            'error': str(e)
        }

def analyze_semantic_shifts(parallel_pairs):
    """Analyze semantic shifts between versions."""
    semantic_changes = {
        'meaning_preservation': [],
        'semantic_fields': defaultdict(int),
        'shift_patterns': defaultdict(int)
    }
    
    for pair in parallel_pairs:
        # Analyze semantic changes
        changes = identify_semantic_changes(pair['original'], pair['translation'])
        semantic_changes['meaning_preservation'].append(changes['preservation_score'])
        
        # Update semantic field changes
        for field, count in changes['field_changes'].items():
            semantic_changes['semantic_fields'][field] += count
    
    return semantic_changes

def analyze_metaphor_preservation(parallel_pairs):
    """Analyze preservation of metaphors between versions."""
    return identify_metaphors(parallel_pairs)

def identify_cultural_adaptations(parallel_pairs):
    """Identify cultural adaptations in translations."""
    cultural_elements = {
        'religious': ['allah', 'bismillah', 'elhamdülillah'],
        'social': ['efendi', 'ağa', 'hanım'],
        'traditional': ['bayram', 'ramazan', 'kurban']
    }
    
    adaptations = {
        'preserved': defaultdict(int),
        'modified': defaultdict(int),
        'removed': defaultdict(int)
    }
    
    for pair in parallel_pairs:
        for category, elements in cultural_elements.items():
            for element in elements:
                if element in pair['original'].lower():
                    if element in pair['translation'].lower():
                        adaptations['preserved'][category] += 1
                    else:
                        adaptations['removed'][category] += 1
    
    return dict(adaptations)

def analyze_diachronic_changes(parallel_pairs):
    """Analyze changes between original and modern versions."""
    try:
        return {
            'lexical_changes': {
                'retention_rate': calculate_lexical_retention(parallel_pairs),
                'modernization_patterns': identify_modernization_patterns(parallel_pairs),
                'archaic_terms': identify_archaic_terms(parallel_pairs)
            },
            'syntactic_changes': {
                'word_order': analyze_word_order_changes(parallel_pairs),
                'clause_structure': analyze_clause_structure(parallel_pairs),
                'complexity_comparison': compare_syntactic_complexity(parallel_pairs)
            },
            'semantic_preservation': {
                'meaning_shifts': analyze_semantic_shifts(parallel_pairs),
                'metaphor_preservation': analyze_metaphor_preservation(parallel_pairs),
                'cultural_adaptations': identify_cultural_adaptations(parallel_pairs)
            }
        }
    except Exception as e:
        print(f"Error in diachronic analysis: {e}")
        return {
            'lexical_changes': {'retention_rate': 0, 'modernization_patterns': {}, 'archaic_terms': {}},
            'syntactic_changes': {'word_order': {}, 'clause_structure': 0, 'complexity_comparison': {}},
            'semantic_preservation': {'meaning_shifts': {}, 'metaphor_preservation': {}, 'cultural_adaptations': {}}
        }

def analyze_dialectal_features(parallel_pairs):
    """Analyze dialectal characteristics."""
    return {
        'phonological': identify_phonological_patterns(parallel_pairs),
        'morphological': identify_morphological_patterns(parallel_pairs),
        'lexical': {
            'dialect_specific_terms': identify_dialect_terms(parallel_pairs),
            'regional_variations': analyze_regional_variations(parallel_pairs)
        }
    }

def analyze_style(parallel_pairs):
    """Analyze stylistic features."""
    return {
        'poetic_features': analyze_poetic_features(parallel_pairs),
        'register': {
            'formality_level': assess_formality(parallel_pairs),
            'register_shifts': identify_register_shifts(parallel_pairs)
        },
        'rhetorical_devices': identify_rhetorical_devices(parallel_pairs)
    }

def calculate_lexical_retention(parallel_pairs):
    """Calculate how many words are retained in modern version."""
    total_retention = 0
    for pair in parallel_pairs:
        original_words = set(word.lower() for word in word_tokenize(pair['original']))
        modern_words = set(word.lower() for word in word_tokenize(pair['translation']))
        retention = len(original_words & modern_words) / len(original_words)
        total_retention += retention
    return total_retention / len(parallel_pairs)

def analyze_word_order_changes(parallel_pairs):
    """Analyze changes in word order between original and translation."""
    changes = {
        'sov_to_svo': 0,  # Subject-Object-Verb to Subject-Verb-Object
        'word_position_shifts': [],
        'average_position_change': 0.0
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        trans_words = word_tokenize(pair['translation'].lower())
        
        # Calculate position shifts for shared words
        shared_words = set(orig_words) & set(trans_words)
        position_shifts = []
        
        for word in shared_words:
            orig_pos = orig_words.index(word)
            trans_pos = trans_words.index(word)
            relative_shift = abs(orig_pos / len(orig_words) - trans_pos / len(trans_words))
            position_shifts.append(relative_shift)
        
        if position_shifts:
            changes['word_position_shifts'].extend(position_shifts)
            
        # Check for SOV to SVO transformation
        if is_sov_to_svo(orig_words, trans_words):
            changes['sov_to_svo'] += 1
    
    # Calculate average position change
    if changes['word_position_shifts']:
        changes['average_position_change'] = np.mean(changes['word_position_shifts'])
    
    return changes

def is_sov_to_svo(orig_words, trans_words):
    """Detect if there's a transformation from SOV to SVO order."""
    # Simple heuristic: check if last word in original (likely verb)
    # appears earlier in translation
    if len(orig_words) > 0 and len(trans_words) > 0:
        last_word = orig_words[-1]
        if last_word in trans_words:
            trans_pos = trans_words.index(last_word)
            return trans_pos < len(trans_words) - 1
    return False

def compare_syntactic_complexity(parallel_pairs):
    """Compare syntactic complexity between versions."""
    try:
        complexity_metrics = {
            'length_metrics': {
                'average_sentence_length_ratio': np.mean([
                    len(p['translation'].split()) / len(p['original'].split())
                    for p in parallel_pairs
                ]),
                'variance_in_length_ratio': np.std([
                    len(p['translation'].split()) / len(p['original'].split())
                    for p in parallel_pairs
                ])
            },
            'word_order': analyze_word_order_changes(parallel_pairs),
            'structural_changes': {
                'average_clause_count_ratio': analyze_clause_structure(parallel_pairs),
                'morphological_complexity': compare_morphological_complexity(parallel_pairs)
            }
        }
        return complexity_metrics
    except Exception as e:
        print(f"Error in syntactic complexity analysis: {e}")
        return {
            'length_metrics': {'average_sentence_length_ratio': 0, 'variance_in_length_ratio': 0},
            'word_order': {'sov_to_svo': 0, 'average_position_change': 0},
            'structural_changes': {'average_clause_count_ratio': 0, 'morphological_complexity': 0}
        }

def analyze_clause_structure(parallel_pairs):
    """Analyze changes in clause structure."""
    try:
        # Simple heuristic: count potential clause markers
        clause_markers = {'ki', 've', 'ama', 'fakat', 'lakin', 'çünkü', 'zira', 'eğer'}
        
        ratios = []
        for pair in parallel_pairs:
            orig_clauses = sum(1 for word in word_tokenize(pair['original'].lower()) 
                             if word in clause_markers) + 1
            trans_clauses = sum(1 for word in word_tokenize(pair['translation'].lower()) 
                              if word in clause_markers) + 1
            ratios.append(trans_clauses / orig_clauses if orig_clauses > 0 else 1)
        
        return np.mean(ratios) if ratios else 1.0
    except Exception:
        return 1.0

def compare_morphological_complexity(parallel_pairs):
    """Compare morphological complexity between versions."""
    try:
        # Simple heuristic: average word length ratio
        orig_avg_lengths = []
        trans_avg_lengths = []
        
        for pair in parallel_pairs:
            orig_words = word_tokenize(pair['original'].lower())
            trans_words = word_tokenize(pair['translation'].lower())
            
            if orig_words and trans_words:
                orig_avg = np.mean([len(word) for word in orig_words])
                trans_avg = np.mean([len(word) for word in trans_words])
                orig_avg_lengths.append(orig_avg)
                trans_avg_lengths.append(trans_avg)
        
        if orig_avg_lengths and trans_avg_lengths:
            return np.mean(trans_avg_lengths) / np.mean(orig_avg_lengths)
        return 1.0
    except Exception:
        return 1.0

def is_poetry(text):
    """Detect if text is poetry based on common patterns."""
    # Check for common poetry markers
    poetry_markers = [
        'Müfteilün', 'Fâilün', 'Mefâilün', 'Feilâtün',  # Common aruz meters
        '###',  # Header markers for verses
        '\n\n',  # Multiple line breaks between stanzas
    ]
    
    return any(marker in text for marker in poetry_markers)

def extract_meters(text):
    """Extract metrical patterns from Ottoman poetry."""
    # Common aruz patterns in Ottoman poetry
    aruz_patterns = {
        'fâilâtün_fâilâtün_fâilün': {
            'pattern': '- + - - / - + - - / - + -',
            'syllables': 11
        },
        'mefâîlün_mefâîlün_feûlün': {
            'pattern': '+ - - - / + - - - / + - -',
            'syllables': 11
        },
        'mefûlü_mefâîlü_feûlün': {
            'pattern': '- - + / + - - + / + - -',
            'syllables': 10
        },
        'feilâtün_feilâtün_feilün': {
            'pattern': '+ + - - / + + - - / + + -',
            'syllables': 11
        }
    }
    
    # Hece vezni patterns
    hece_patterns = {
        '7_hece': 7,
        '8_hece': 8,
        '11_hece': 11,
        '14_hece': 14
    }
    
    def count_syllables(line):
        """Count syllables in a line of Ottoman Turkish."""
        vowels = set('aâeêıîiîoôöuûü')
        count = 0
        prev_char = ''
        
        for char in line.lower():
            if char in vowels:
                # Handle special cases of diphthongs and adjacent vowels
                if prev_char not in vowels:
                    count += 1
            prev_char = char
        return count
    
    def match_aruz_pattern(line):
        """Match line against aruz patterns."""
        syllable_count = count_syllables(line)
        potential_matches = []
        
        for meter_name, meter_info in aruz_patterns.items():
            if syllable_count == meter_info['syllables']:
                # Convert line to metrical pattern
                pattern = convert_to_pattern(line)
                if pattern_similarity(pattern, meter_info['pattern']) > 0.8:
                    potential_matches.append(meter_name)
        
        return potential_matches
    
    def convert_to_pattern(line):
        """Convert line to metrical pattern of long (-) and short (+) syllables."""
        pattern = []
        vowels = set('aâeêıîiîoôöuûü')
        consonants = set('bcçdfgğhjklmnprsştvyzqwx')
        
        words = line.lower().split()
        for word in words:
            i = 0
            while i < len(word):
                if word[i] in vowels:
                    # Check for long syllable conditions
                    if (i + 1 < len(word) and word[i + 1] in consonants) or \
                       (i + 1 == len(word)) or \
                       (word[i] in 'âêîôû'):
                        pattern.append('-')  # Long syllable
                    else:
                        pattern.append('+')  # Short syllable
                i += 1
        
        return ' '.join(pattern)
    
    def pattern_similarity(pattern1, pattern2):
        """Calculate similarity between two metrical patterns."""
        # Remove spaces and compare directly
        p1 = pattern1.replace(' ', '')
        p2 = pattern2.replace(' ', '')
        
        if len(p1) != len(p2):
            return 0.0
            
        matches = sum(1 for a, b in zip(p1, p2) if a == b)
        return matches / len(p1)
    
    # Analyze the text
    meters_found = {
        'aruz': defaultdict(int),
        'hece': defaultdict(int),
        'unidentified': 0
    }
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for aruz meters
        aruz_matches = match_aruz_pattern(line)
        if aruz_matches:
            for meter in aruz_matches:
                meters_found['aruz'][meter] += 1
        else:
            # Check for hece vezni
            syllable_count = count_syllables(line)
            for pattern_name, pattern_count in hece_patterns.items():
                if syllable_count == pattern_count:
                    meters_found['hece'][pattern_name] += 1
                    break
            else:
                meters_found['unidentified'] += 1
    
    return {
        'meters': dict(meters_found['aruz']),
        'hece_patterns': dict(meters_found['hece']),
        'unidentified_lines': meters_found['unidentified'],
        'total_lines': len(lines)
    }

def analyze_meter_preservation(parallel_pairs):
    """Analyze preservation of poetic meter."""
    try:
        if not parallel_pairs:
            return {
                'original_meters': {},
                'preserved_ratio': 0.0,
                'meter_changes': {}
            }
        
        # Analyze meters in original and translation
        original_meters = extract_meters(parallel_pairs[0]['original'])
        translation_meters = extract_meters(parallel_pairs[0]['translation'])
        
        # Calculate preservation ratio
        total_original_patterns = sum(original_meters['meters'].values()) + \
                                sum(original_meters['hece_patterns'].values())
        total_preserved_patterns = sum(
            min(original_meters['meters'].get(k, 0), translation_meters['meters'].get(k, 0))
            for k in set(original_meters['meters']) & set(translation_meters['meters'])
        )
        
        preservation_ratio = total_preserved_patterns / total_original_patterns if total_original_patterns > 0 else 0.0
        
        return {
            'original_meters': original_meters,
            'translation_meters': translation_meters,
            'preserved_ratio': preservation_ratio,
            'meter_changes': {
                'patterns_lost': set(original_meters['meters']) - set(translation_meters['meters']),
                'patterns_added': set(translation_meters['meters']) - set(original_meters['meters']),
                'patterns_preserved': set(original_meters['meters']) & set(translation_meters['meters'])
            }
        }
        
    except Exception as e:
        print(f"Error in meter preservation analysis: {e}")
        return {
            'original_meters': {},
            'preserved_ratio': 0.0,
            'error': str(e)
        }

def analyze_rhyme_patterns(parallel_pairs):
    """Analyze rhyme patterns in parallel texts."""
    return {
        'rhyme_scheme': 'Unknown',  # Placeholder for actual rhyme analysis
        'rhyme_preserved': False
    }

def analyze_metaphor_preservation(parallel_pairs):
    """Analyze preservation of metaphorical elements."""
    return {
        'metaphor_count': 0,  # Placeholder for actual metaphor analysis
        'preservation_rate': 0.0
    }

def analyze_poetic_features(parallel_pairs):
    """Analyze poetic features if text is poetry."""
    return {
        'meter_preservation': analyze_meter_preservation(parallel_pairs),
        'rhyme_patterns': analyze_rhyme_patterns(parallel_pairs),
        'metaphor_preservation': analyze_metaphor_preservation(parallel_pairs)
    }

def calculate_morphological_complexity(texts):
    """Calculate morphological complexity of texts."""
    # Ottoman Turkish suffixes and morphological markers
    suffixes = {
        'case_markers': ['de', 'da', 'den', 'dan', 'e', 'a', 'i', 'ı', 'u', 'ü'],
        'possessive': ['im', 'ım', 'um', 'üm', 'in', 'ın', 'un', 'ün'],
        'plural': ['ler', 'lar'],
        'verbal': ['mek', 'mak', 'miş', 'mış', 'muş', 'müş', 'di', 'dı', 'du', 'dü']
    }
    
    complexity_scores = []
    
    for text in texts:
        words = word_tokenize(text.lower())
        if not words:
            continue
            
        # Count morphological markers per word
        morpheme_counts = []
        for word in words:
            count = 0
            for category in suffixes.values():
                for suffix in category:
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

def identify_style_markers(texts, text_type):
    """Identify style markers in texts."""
    # Ottoman Turkish specific style markers
    style_markers = {
        'persian_compounds': ['name', 'hane', 'dar', 'kar', 'zade', 'perest'],
        'arabic_patterns': ['et', 'ol', 'eyle', 'kıl'],
        'honorifics': ['hazret', 'efendi', 'bey', 'hanım', 'ağa', 'paşa'],
        'poetic_markers': ['ey', 'hey', 'ah', 'oh', 'şol', 'kim', 'ki']
    }
    
    markers_found = defaultdict(int)
    register_scores = []
    formality_markers = []
    
    for text in texts:
        words = word_tokenize(text.lower())
        
        # Count style markers
        for category, markers in style_markers.items():
            for marker in markers:
                markers_found[category] += sum(1 for word in words if marker in word.lower())
        
        # Calculate register score (formal vs informal)
        register_score = calculate_register_score(words)
        register_scores.append(register_score)
        
        # Identify formality markers
        formality_markers.extend(identify_formality_markers(words))
    
    return {
        'style_markers_frequency': dict(markers_found),
        'average_register_score': np.mean(register_scores) if register_scores else 0.0,
        'formality_markers': Counter(formality_markers).most_common(10),
        'text_type_specific': analyze_text_type_features(texts, text_type)
    }

def calculate_register_score(words):
    """Calculate formality register score."""
    formal_markers = {'efendi', 'hazret', 'buyur', 'teşrif', 'istirham', 'rica'}
    informal_markers = {'be', 'ulan', 'yahu', 'hele', 'haydi'}
    
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

def analyze_text_type_features(texts, text_type):
    """Analyze features specific to text type (poetry/prose)."""
    if text_type == 'poetry':
        return analyze_poetic_features(texts)
    else:
        return analyze_prose_features(texts)

def analyze_poetic_features(texts):
    """Analyze poetic features."""
    return {
        'meter_patterns': identify_meter_patterns(texts),
        'rhyme_schemes': analyze_rhyme_schemes(texts),
        'poetic_devices': identify_poetic_devices(texts)
    }

def analyze_prose_features(texts):
    """Analyze prose features."""
    return {
        'paragraph_structure': analyze_paragraph_structure(texts),
        'narrative_markers': identify_narrative_markers(texts),
        'discourse_connectors': analyze_discourse_connectors(texts)
    }

def analyze_semantic_shifts(parallel_pairs):
    """Analyze semantic changes between versions."""
    semantic_changes = {
        'meaning_preservation': [],
        'semantic_shifts': [],
        'context_changes': []
    }
    
    for pair in parallel_pairs:
        # Analyze each sentence pair
        orig_sents = pair['original'].split('.')
        trans_sents = pair['translation'].split('.')
        
        for orig, trans in zip(orig_sents, trans_sents):
            if not orig.strip() or not trans.strip():
                continue
                
            similarity = calculate_similarity(orig, trans)
            if similarity > 0.8:
                semantic_changes['meaning_preservation'].append((orig, trans))
            elif similarity > 0.5:
                semantic_changes['semantic_shifts'].append((orig, trans))
            else:
                semantic_changes['context_changes'].append((orig, trans))
    
    return {
        'preservation_rate': len(semantic_changes['meaning_preservation']) / len(parallel_pairs),
        'shift_rate': len(semantic_changes['semantic_shifts']) / len(parallel_pairs),
        'major_changes': len(semantic_changes['context_changes']) / len(parallel_pairs),
        'examples': {
            'preserved': semantic_changes['meaning_preservation'][:3],
            'shifted': semantic_changes['semantic_shifts'][:3],
            'changed': semantic_changes['context_changes'][:3]
        }
    }

def suggest_research_applications(parallel_pairs):
    """Suggest research applications for the analyzed texts."""
    # Implementation of research application suggestion
    # This can be based on various features such as word frequency, sentence length, etc.
    pass

def identify_limitations(parallel_pairs):
    """Identify limitations of the analyzed texts."""
    # Implementation of limitation identification
    # This can be based on various features such as word frequency, sentence length, etc.
    pass

def prepare_diachronic_visualization(metadata):
    """Prepare data for diachronic visualization."""
    # Implementation of diachronic visualization preparation
    # This can be based on various features such as word frequency, sentence length, etc.
    pass

def prepare_dialectal_visualization(metadata):
    """Prepare data for dialectal visualization."""
    # Implementation of dialectal visualization preparation
    # This can be based on various features such as word frequency, sentence length, etc.
    pass

def prepare_statistical_visualization(metadata):
    """Prepare data for statistical visualization."""
    # Implementation of statistical visualization preparation
    # This can be based on various features such as word frequency, sentence length, etc.
    pass

def save_parallel_formats(parallel_pairs, output_folder):
    """Save parallel texts in multiple formats."""
    # Implementation of parallel text saving in multiple formats
    # This can be based on various formats such as Markdown, HTML, PDF, etc.
    pass

def generate_research_report(metadata, output_folder):
    """Generate a research report for the analyzed texts."""
    # Implementation of research report generation
    # This can be based on various features such as word frequency, sentence length, etc.
    pass

def identify_modernization_patterns(parallel_pairs):
    """Identify patterns in modernization of words and phrases."""
    patterns = {
        'orthographic_changes': [],
        'lexical_replacements': [],
        'common_substitutions': defaultdict(int)
    }
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        trans_words = word_tokenize(pair['translation'].lower())
        
        # Track common word replacements
        for orig, modern in zip(orig_words, trans_words):
            if orig != modern:
                patterns['common_substitutions'][(orig, modern)] += 1
                
                # Check for systematic orthographic changes
                if calculate_similarity(orig, modern) > 0.7:
                    patterns['orthographic_changes'].append((orig, modern))
                else:
                    patterns['lexical_replacements'].append((orig, modern))
    
    # Convert to frequency dictionaries
    patterns['common_substitutions'] = dict(sorted(
        patterns['common_substitutions'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:20])  # Top 20 most common substitutions
    
    return patterns

def identify_archaic_terms(parallel_pairs):
    """Identify archaic terms and their modern equivalents."""
    archaic_terms = defaultdict(list)
    
    for pair in parallel_pairs:
        orig_words = word_tokenize(pair['original'].lower())
        trans_words = word_tokenize(pair['translation'].lower())
        
        # Words present in original but not in translation
        unique_to_original = set(orig_words) - set(trans_words)
        
        for word in unique_to_original:
            # Find potential modern equivalent using context
            context = find_context_equivalent(word, pair['original'], pair['translation'])
            if context:
                archaic_terms[word].append(context)
    
    return dict(archaic_terms)

def find_context_equivalent(word, original, translation):
    """Find modern equivalent of an archaic word using context."""
    # Get word context in original text
    orig_words = word_tokenize(original.lower())
    trans_words = word_tokenize(translation.lower())
    
    try:
        word_idx = orig_words.index(word)
        context_start = max(0, word_idx - 2)
        context_end = min(len(orig_words), word_idx + 3)
        
        # Get corresponding context in translation
        context_length = context_end - context_start
        trans_start = max(0, word_idx - 2)
        trans_end = min(len(trans_words), trans_start + context_length)
        
        return {
            'original_context': ' '.join(orig_words[context_start:context_end]),
            'translation_context': ' '.join(trans_words[trans_start:trans_end]),
            'likely_equivalent': find_most_similar_word(word, trans_words[trans_start:trans_end])
        }
    except ValueError:
        return None

def find_most_similar_word(word, candidates):
    """Find the most similar word from candidates."""
    if not candidates:
        return None
    
    similarities = [(w, calculate_similarity(word, w)) for w in candidates]
    return max(similarities, key=lambda x: x[1])[0]

def identify_meter_patterns(texts):
    """Identify metrical patterns in poetic texts."""
    meter_patterns = {
        'aruz': {
            'fâilâtün': r'[-+][-+][-+][-]',
            'mefâîlün': r'[+][-][-+][-]',
            'müstef\'ilün': r'[-][-][+][-]'
        },
        'hece': {
            '7_hece': r'.{7}',
            '11_hece': r'.{11}'
        }
    }
    
    results = {
        'identified_meters': defaultdict(int),
        'line_analysis': []
    }
    
    for text in texts:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for aruz patterns
            for meter_name, pattern in meter_patterns['aruz'].items():
                if re.search(pattern, line):
                    results['identified_meters'][meter_name] += 1
            
            # Check for hece patterns
            syllable_count = count_syllables(line)
            if syllable_count in [7, 11]:
                results['identified_meters'][f'{syllable_count}_hece'] += 1
                
            results['line_analysis'].append({
                'line': line,
                'syllables': syllable_count
            })
    
    return dict(results)

def analyze_rhyme_schemes(texts):
    """Analyze rhyme patterns in texts."""
    rhyme_patterns = defaultdict(int)
    stanza_patterns = []
    
    for text in texts:
        stanzas = text.split('\n\n')
        for stanza in stanzas:
            lines = [line.strip() for line in stanza.split('\n') if line.strip()]
            if len(lines) < 2:
                continue
                
            # Get last words for rhyme analysis
            last_words = [line.split()[-1] if line.split() else '' for line in lines]
            rhyme_pattern = identify_rhyme_pattern(last_words)
            rhyme_patterns[rhyme_pattern] += 1
            
            stanza_patterns.append({
                'pattern': rhyme_pattern,
                'lines': len(lines)
            })
    
    return {
        'common_patterns': dict(rhyme_patterns.most_common(5)),
        'stanza_analysis': stanza_patterns[:5]
    }

def identify_poetic_devices(texts):
    """Identify poetic devices in texts."""
    devices = {
        'alliteration': defaultdict(int),
        'assonance': defaultdict(int),
        'redif': [],
        'metaphors': set()
    }
    
    for text in texts:
        lines = text.split('\n')
        for line in lines:
            # Check for alliteration
            words = word_tokenize(line)
            if len(words) >= 2:
                for i in range(len(words)-1):
                    if words[i][0] == words[i+1][0]:
                        devices['alliteration'][words[i][0]] += 1
            
            # Check for redif (repeated phrases at line ends)
            if len(lines) >= 2:
                last_words = [l.split()[-1] if l.split() else '' for l in lines]
                repeated = [w for w, c in Counter(last_words).items() if c > 1]
                devices['redif'].extend(repeated)
    
    return {
        'alliteration': dict(devices['alliteration'].most_common(5)),
        'redif': list(set(devices['redif']))[:5]
    }

def analyze_paragraph_structure(texts):
    """Analyze paragraph structure in prose texts."""
    return {
        'average_length': np.mean([len(para.split()) 
            for text in texts for para in text.split('\n\n')]),
        'sentence_distribution': analyze_sentence_distribution(texts),
        'paragraph_transitions': identify_paragraph_transitions(texts)
    }

def identify_narrative_markers(texts):
    """Identify narrative markers in prose texts."""
    narrative_markers = {
        'temporal': ['sonra', 'önce', 'evvel', 'şimdi', 'vaktiyle'],
        'causal': ['çünkü', 'zira', 'bu sebeple', 'dolayısıyla'],
        'spatial': ['orada', 'burada', 'şurada', 'ötede']
    }
    
    found_markers = defaultdict(list)
    for text in texts:
        for category, markers in narrative_markers.items():
            for marker in markers:
                if marker in text.lower():
                    found_markers[category].append(marker)
    
    return dict(found_markers)

def analyze_discourse_connectors(texts):
    """Analyze discourse connectors in texts."""
    connectors = {
        'additive': ['ve', 'ile', 'hem', 'de'],
        'adversative': ['ama', 'fakat', 'lakin', 'ancak'],
        'causal': ['çünkü', 'zira', 'dolayısıyla'],
        'temporal': ['sonra', 'önce', 'evvel']
    }
    
    connector_counts = defaultdict(int)
    for text in texts:
        words = word_tokenize(text.lower())
        for category, conn_list in connectors.items():
            for connector in conn_list:
                connector_counts[f"{category}_{connector}"] += words.count(connector)
    
    return dict(connector_counts)

def count_syllables(text):
    """Count syllables in Ottoman Turkish text."""
    vowels = set('aâeêıîiîoôöuûü')
    count = 0
    for char in text.lower():
        if char in vowels:
            count += 1
    return count

def identify_rhyme_pattern(words):
    """Identify rhyme pattern in a list of words."""
    if not words:
        return ''
        
    pattern = []
    rhyme_map = {}
    current_letter = 'a'
    
    for word in words:
        rhyme_sound = get_rhyme_sound(word)
        if rhyme_sound not in rhyme_map:
            rhyme_map[rhyme_sound] = current_letter
            current_letter = chr(ord(current_letter) + 1)
        pattern.append(rhyme_map[rhyme_sound])
    
    return ''.join(pattern)

def get_rhyme_sound(word):
    """Get the rhyming sound of a word."""
    if not word:
        return ''
    # Simple implementation - last syllable
    vowels = 'aâeêıîiîoôöuûü'
    word = word.lower()
    last_vowel_index = max((word.rindex(v) for v in vowels if v in word), default=-1)
    if last_vowel_index == -1:
        return word
    return word[last_vowel_index:]

def analyze_sentence_distribution(texts):
    """Analyze sentence length distribution in texts."""
    sentence_lengths = [len(sent.split()) 
        for text in texts 
        for sent in text.split('.') 
        if sent.strip()]
    
    return {
        'average': np.mean(sentence_lengths) if sentence_lengths else 0,
        'std_dev': np.std(sentence_lengths) if sentence_lengths else 0,
        'distribution': Counter(sentence_lengths)
    }

def identify_paragraph_transitions(texts):
    """Identify transition words between paragraphs."""
    transition_words = {
        'addition': ['ayrıca', 'dahası', 'üstelik'],
        'contrast': ['fakat', 'lakin', 'ancak'],
        'conclusion': ['sonuç olarak', 'netice itibariyle']
    }
    
    transitions_found = defaultdict(list)
    for text in texts:
        paragraphs = text.split('\n\n')
        for i in range(1, len(paragraphs)):
            first_words = paragraphs[i].split()[:3]
            first_phrase = ' '.join(first_words).lower()
            
            for category, words in transition_words.items():
                if any(word in first_phrase for word in words):
                    transitions_found[category].append(first_phrase)
    
    return dict(transitions_found)

class ParallelTextVisualizer:
    """Visualizer for parallel text analysis."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn')
        
    def create_all_visualizations(self, metadata):
        """Generate all visualizations from metadata."""
        try:
            self._create_text_length_comparison(metadata)
            self._create_similarity_distribution(metadata)
            self._create_feature_comparison(metadata)
            self._save_visualization_data(metadata)
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def _create_text_length_comparison(self, metadata):
        """Create bar chart comparing original and translation lengths."""
        try:
            if 'parallel_texts' not in metadata:
                return
                
            pairs = metadata['parallel_texts']
            if not pairs:
                return
                
            # Extract lengths
            orig_lengths = [len(p['original'].split()) for p in pairs]
            trans_lengths = [len(p['translation'].split()) for p in pairs]
            
            plt.figure(figsize=(10, 6))
            plt.bar(['Original', 'Translation'], 
                   [np.mean(orig_lengths), np.mean(trans_lengths)],
                   yerr=[np.std(orig_lengths), np.std(trans_lengths)])
            plt.title('Average Text Length Comparison')
            plt.ylabel('Number of Words')
            plt.savefig(self.viz_dir / 'length_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in length comparison visualization: {e}")
    
    def _create_similarity_distribution(self, metadata):
        """Create histogram of similarity scores."""
        try:
            if 'parallel_texts' not in metadata:
                return
                
            pairs = metadata['parallel_texts']
            if not pairs:
                return
                
            # Extract similarity scores
            scores = [p.get('similarity_score', 0) for p in pairs]
            
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=20, edgecolor='black')
            plt.title('Distribution of Similarity Scores')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.savefig(self.viz_dir / 'similarity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in similarity distribution visualization: {e}")
    
    def _create_feature_comparison(self, metadata):
        """Create radar chart of linguistic features."""
        try:
            if 'linguistic_features' not in metadata:
                return
                
            features = metadata.get('linguistic_features', {})
            if not features:
                return
                
            # Extract feature values
            categories = ['Lexical', 'Syntactic', 'Semantic']
            orig_values = [
                features.get('original', {}).get('vocabulary', {}).get('lexical_diversity', 0),
                features.get('original', {}).get('structural', {}).get('complexity', 0),
                features.get('original', {}).get('semantic', {}).get('preservation', 0)
            ]
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, orig_values)
            ax.fill(angles, orig_values, alpha=0.25)
            ax.set_xticks(angles)
            ax.set_xticklabels(categories)
            plt.title('Linguistic Feature Comparison')
            
            plt.savefig(self.viz_dir / 'feature_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in feature comparison visualization: {e}")
    
    def _save_visualization_data(self, metadata):
        """Save visualization data in JSON format for potential reuse."""
        try:
            viz_data = {
                'metadata_timestamp': datetime.datetime.now().isoformat(),
                'visualizations_created': [
                    'length_comparison.png',
                    'similarity_distribution.png',
                    'feature_comparison.png'
                ],
                'data_summary': {
                    'total_pairs': len(metadata.get('parallel_texts', [])),
                    'features_analyzed': list(metadata.get('linguistic_features', {}).keys())
                }
            }
            
            with open(self.viz_dir / 'visualization_data.json', 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving visualization data: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_markdown.py <path_to_input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    try:
        stats = process_directory(input_directory)
        print(f"\nAll PDF files in {input_directory} have been processed.")
        print("Markdown files are saved in: C:\\Users\\Administrator\\Desktop\\cook\\Ottoman-NLP\\corpus-texts\\md_out")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()














