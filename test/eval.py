import re
from collections import Counter
from pathlib import Path
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr
from collections import defaultdict
import math
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import traceback

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_multiple_texts(directory):
    texts = []
    file_paths = list(directory.glob('*.txt'))
    for file_path in tqdm(file_paths, desc="Loading texts"):
        texts.append(load_text(file_path))
    return ' '.join(texts)

def normalize_chars(text):
    char_map = {
        'â': 'a', 'Â': 'A', 'î': 'i', 'Î': 'I', 'û': 'u', 'Û': 'U',
        'ğ': 'g', 'Ğ': 'G', 'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S', 'ü': 'u', 'Ü': 'U', 'ç': 'c', 'Ç': 'C'
    }
    for old, new in char_map.items():
        text = text.replace(old, new)
    return text

def preprocess_text(text):
    text = normalize_chars(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stopwords = set(['-i', 'ki', 'i', 'a', 've', 'bu', 'bir', 'da', 'de', 'ile'])
    tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
    return tokens

class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

def calculate_perplexity_gpu(tokens, n=2, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")
    dataset = TextDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    n_grams = defaultdict(lambda: defaultdict(int))
    for batch in tqdm(dataloader, desc="Calculating n-grams"):
        for i in range(len(batch) - n + 1):
            context = tuple(batch[i:i+n-1])
            token = batch[i+n-1]
            n_grams[context][token] += 1

    log_prob = 0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="Calculating perplexity"):
        batch_tensor = torch.tensor([ord(c) for c in batch]).to(device)
        for i in range(n-1, len(batch_tensor)):
            context = tuple(batch_tensor[i-n+1:i].tolist())
            token = batch_tensor[i].item()
            count = n_grams[context][token]
            total = sum(n_grams[context].values())
            prob = count / total if total > 0 else 1e-10
            log_prob += math.log2(prob)
        total_tokens += len(batch_tensor) - n + 1

    perplexity = 2 ** (-log_prob / total_tokens)
    return perplexity

def calculate_oov(original_tokens, processed_tokens):
    original_vocab = set(original_tokens)
    processed_vocab = set(processed_tokens)
    oov_tokens = processed_vocab - original_vocab
    oov_rate = len(oov_tokens) / len(processed_vocab)
    return oov_rate, oov_tokens

def calculate_metrics(original, processed):
    print("Starting metric calculations...")
    original_tokens = preprocess_text(original)
    processed_tokens = preprocess_text(processed)
    print(f"Preprocessed tokens: Original {len(original_tokens)}, Processed {len(processed_tokens)}")

    original_count = len(original_tokens)
    processed_count = len(processed_tokens)
    
    original_vocab = len(set(original_tokens))
    processed_vocab = len(set(processed_tokens))
    original_char_count = len(original)
    processed_char_count = len(processed)
    
    original_sentences = len(re.findall(r'\.\s', original))
    processed_sentences = len(re.findall(r'\.\s', processed))
    lev_distance = Levenshtein.distance(original, processed)
    
    original_freq = Counter(original_tokens)
    processed_freq = Counter(processed_tokens)
    
    tfidf = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = tfidf.fit_transform([original, processed])
    feature_names = tfidf.get_feature_names_out()
    original_tfidf = dict(zip(feature_names, tfidf_matrix[0].toarray()[0]))
    processed_tfidf = dict(zip(feature_names, tfidf_matrix[1].toarray()[0]))
    
    common_words = set(original_freq.keys()) & set(processed_freq.keys())
    original_freqs = [original_freq[word] for word in common_words]
    processed_freqs = [processed_freq[word] for word in common_words]
    spearman_corr, _ = spearmanr(original_freqs, processed_freqs)
    
    print("Calculating OOV rate...")
    oov_rate, oov_tokens = calculate_oov(original_tokens, processed_tokens)
    print("Calculating original perplexity...")
    original_perplexity = calculate_perplexity_gpu(original_tokens)
    print("Calculating processed perplexity...")
    processed_perplexity = calculate_perplexity_gpu(processed_tokens)
    
    print("Metric calculations complete.")
    return {
        'original_tokens': original_count,
        'processed_tokens': processed_count,
        'original_vocab': original_vocab,
        'processed_vocab': processed_vocab,
        'original_chars': original_char_count,
        'processed_chars': processed_char_count,
        'original_sentences': original_sentences,
        'processed_sentences': processed_sentences,
        'levenshtein_distance': lev_distance,
        'original_freq': original_freq,
        'processed_freq': processed_freq,
        'original_tfidf': original_tfidf,
        'processed_tfidf': processed_tfidf,
        'spearman_correlation': spearman_corr,
        'oov_rate': oov_rate,
        'oov_tokens': oov_tokens,
        'original_perplexity': original_perplexity,
        'processed_perplexity': processed_perplexity
    }

def evaluate_corpus(original_file, processed_dir):
    try:
        if not original_file.exists():
            print(f"Error: Original file does not exist.")
            return None

        print("Loading original text...")
        original_text = load_text(original_file)
        print("Loading processed texts...")
        processed_text = load_multiple_texts(processed_dir)
        
        print("Calculating metrics...")
        metrics = calculate_metrics(original_text, processed_text)
        return {"Corpus Analysis": metrics}
    except Exception as e:
        print(f"An error occurred during corpus evaluation: {str(e)}")
        traceback.print_exc()
        return None

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename, metrics in results.items():
            f.write(f"Results for {filename}:\n")
            f.write(f"  Original tokens: {metrics['original_tokens']}\n")
            f.write(f"  Processed tokens: {metrics['processed_tokens']}\n")
            f.write(f"  Original vocabulary: {metrics['original_vocab']}\n")
            f.write(f"  Processed vocabulary: {metrics['processed_vocab']}\n")
            f.write(f"  Original characters: {metrics['original_chars']}\n")
            f.write(f"  Processed characters: {metrics['processed_chars']}\n")
            f.write(f"  Original sentences: {metrics['original_sentences']}\n")
            f.write(f"  Processed sentences: {metrics['processed_sentences']}\n")
            f.write(f"  Levenshtein distance: {metrics['levenshtein_distance']}\n")
            f.write(f"  Spearman correlation: {metrics['spearman_correlation']:.4f}\n")
            f.write(f"  OOV rate: {metrics['oov_rate']:.4f}\n")
            f.write(f"  Original perplexity: {metrics['original_perplexity']:.2f}\n")
            f.write(f"  Processed perplexity: {metrics['processed_perplexity']:.2f}\n")
            f.write(f"  Top 20 original tokens: {metrics['original_freq'].most_common(20)}\n")
            f.write(f"  Top 20 processed tokens: {metrics['processed_freq'].most_common(20)}\n")
            f.write(f"  OOV tokens: {', '.join(list(metrics['oov_tokens'])[:50])}\n\n")

def create_visualizations(results, output_dir):
    for filename, metrics in results.items():
        # Bar chart for token and vocabulary counts
        plt.figure(figsize=(12, 6))
        labels = ['Tokens', 'Vocabulary', 'Characters', 'Sentences']
        original = [metrics['original_tokens'], metrics['original_vocab'], metrics['original_chars'], metrics['original_sentences']]
        processed = [metrics['processed_tokens'], metrics['processed_vocab'], metrics['processed_chars'], metrics['processed_sentences']]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, original, width, label='Original', color='#f0027f')
        plt.bar(x + width/2, processed, width, label='Processed', color='#386cb0')
        plt.xlabel('Metrics')
        plt.ylabel('Counts')
        plt.title('Comparison of Original and Processed Corpus')
        plt.xticks(x, labels)
        plt.legend()
        plt.savefig(output_dir / 'comparison_chart.png')
        plt.close()

        # Scatter plot for TF-IDF comparison (top 100 words)
        plt.figure(figsize=(16, 12))
        common_words = set(metrics['original_tfidf'].keys()) & set(metrics['processed_tfidf'].keys())
        top_words = sorted(common_words, key=lambda w: max(metrics['original_tfidf'][w], metrics['processed_tfidf'][w]), reverse=True)[:100]
        x = [metrics['original_tfidf'][word] for word in top_words]
        y = [metrics['processed_tfidf'][word] for word in top_words]
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('Original TF-IDF')
        plt.ylabel('Processed TF-IDF')
        plt.title('TF-IDF Comparison (Top 100 Words)')
        plt.plot([0, max(x+y)], [0, max(x+y)], 'r--', lw=2)
        for i, word in enumerate(top_words):
            plt.annotate(word, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / 'tfidf_comparison.png', dpi=300)
        plt.close()

        # Horizontal bar chart for top 30 words frequency comparison
        plt.figure(figsize=(14, 12))
        top_words = set([word for word, _ in metrics['original_freq'].most_common(30)] +
                        [word for word, _ in metrics['processed_freq'].most_common(30)])
        y_pos = np.arange(len(top_words))
        original_freq = [metrics['original_freq'][word] for word in top_words]
        processed_freq = [metrics['processed_freq'][word] for word in top_words]
        
        plt.barh(y_pos, original_freq, align='center', alpha=0.5, color='#f0027f', label='Original')
        plt.barh(y_pos, processed_freq, align='center', alpha=0.5, color='#386cb0', label='Processed')
        plt.yticks(y_pos, list(top_words))
        plt.xlabel('Frequency')
        plt.title('Top 30 Words Frequency Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'frequency_comparison.png')
        plt.close()

        # OOV and Perplexity comparison
        plt.figure(figsize=(8, 6))
        labels = ['OOV Rate', 'Perplexity']
        original_values = [0, metrics['original_perplexity']]
        processed_values = [metrics['oov_rate'], metrics['processed_perplexity']]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, original_values, width, label='Original', color='#f0027f')
        plt.bar(x + width/2, processed_values, width, label='Processed', color='#386cb0')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('OOV Rate and Perplexity Comparison')
        plt.xticks(x, labels)
        plt.legend()
        plt.savefig(output_dir / 'oov_perplexity_comparison.png')
        plt.close()

def load_corpus():
    corpus = {
        'ottoman': [],
        'turkish': [],
        'parallel': []
    }
    
    # Load everythinglm_corrected.txt
    everythinglm_path = root_dir / "corpus-texts" / "datasets" / "everythinglm_corrected.txt"
    if everythinglm_path.exists():
        with open(everythinglm_path, 'r', encoding='utf-8') as f:
            corpus['ottoman'].append(f.read())
    
    # Load clean_txts
    clean_txts_dir = root_dir / "corpus-texts" / "clean_txts"
    corpus['ottoman'].extend(load_and_preprocess_texts(clean_txts_dir))
    
    # Load texts from the 'texts' folder
    texts_dir = root_dir / "corpus-texts" / "texts"
    for subfolder in texts_dir.iterdir():
        if subfolder.is_dir():
            ottoman_path = subfolder / "osmanlıca.txt"
            turkish_path = subfolder / "türkçe.txt"
            if ottoman_path.exists() and turkish_path.exists():
                with open(ottoman_path, 'r', encoding='utf-8') as f:
                    ottoman_text = f.read()
                with open(turkish_path, 'r', encoding='utf-8') as f:
                    turkish_text = f.read()
                corpus['ottoman'].append(ottoman_text)
                corpus['turkish'].append(turkish_text)
                corpus['parallel'].append((ottoman_text, turkish_text))
    
    return corpus

def analyze_parallel_texts(parallel_texts):
    ottoman_lengths = []
    turkish_lengths = []
    length_ratios = []

    for ottoman, turkish in parallel_texts:
        ottoman_words = ottoman.split()
        turkish_words = turkish.split()
        ottoman_lengths.append(len(ottoman_words))
        turkish_lengths.append(len(turkish_words))
        length_ratios.append(len(turkish_words) / len(ottoman_words) if len(ottoman_words) > 0 else 0)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.hist(ottoman_lengths, bins=20, alpha=0.5, label='Ottoman')
    plt.hist(turkish_lengths, bins=20, alpha=0.5, label='Turkish')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths')
    plt.legend()

    plt.subplot(132)
    plt.scatter(ottoman_lengths, turkish_lengths, alpha=0.5)
    plt.xlabel('Ottoman Text Length')
    plt.ylabel('Turkish Text Length')
    plt.title('Ottoman vs Turkish Text Lengths')

    plt.subplot(133)
    plt.hist(length_ratios, bins=20)
    plt.xlabel('Turkish/Ottoman Length Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Length Ratios')

    plt.tight_layout()
    plt.savefig(output_dir / 'parallel_text_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Average Ottoman text length: {np.mean(ottoman_lengths):.2f} words")
    print(f"Average Turkish text length: {np.mean(turkish_lengths):.2f} words")
    print(f"Average Turkish/Ottoman length ratio: {np.mean(length_ratios):.2f}")

def main():
    project_root = Path(__file__).resolve().parents[2]
    original_file = project_root / "corpus-texts" / "test_data" / "test_data.txt"
    processed_dir = project_root / "corpus-texts" / "clean_txts"
    output_dir = project_root / "corpus-texts" / "test_data" / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("Starting corpus analysis...")
    results = evaluate_corpus(original_file, processed_dir)
    if results:
        print("Saving results...")
        save_results(results, output_dir / "corpus_analysis_results.txt")
        print("Creating visualizations...")
        create_visualizations(results, output_dir)
        print(f"Corpus analysis results and visualizations saved in {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
