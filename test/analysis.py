import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import gensim
from gensim.models import FastText
import umap  # Corrected import
from pathlib import Path
import torch  # For GPU support
import datetime

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root_dir = Path(__file__).resolve().parents[2]

def load_and_preprocess_texts(directory):
    texts = []
    for file_path in tqdm(Path(directory).glob('**/*.txt'), desc="Loading texts"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Basic preprocessing: lowercase and remove punctuation
            text = re.sub(r'[^\w\s]', '', text.lower())
            texts.append(text)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return texts

def load_corpus():
    corpus = {
        'ottoman': [],
        'turkish': [],
        'parallel': []
    }
    
    # Load everythinglm_corrected.txt
    everythinglm_path = root_dir / "corpus-texts" / "datasets" / "everythinglm_corrected.txt"
    if everythinglm_path.exists():
        corpus['ottoman'].append(analyze_text_file(everythinglm_path))
    
    # Load clean_txts
    clean_txts_dir = root_dir / "corpus-texts" / "clean_txts"
    corpus['ottoman'].extend([analyze_text_file(f) for f in Path(clean_txts_dir).glob('**/*.txt')])
    
    # Load texts from the 'texts' folder
    texts_dir = root_dir / "corpus-texts" / "texts"
    for subfolder in texts_dir.iterdir():
        if subfolder.is_dir():
            osmanli_path = subfolder / "osmanlıca.txt"
            turkish_path = subfolder / "türkçe.txt"
            if osmanli_path.exists() and turkish_path.exists():
                ottoman_analysis = analyze_text_file(osmanli_path, subfolder.name)
                turkish_analysis = analyze_text_file(turkish_path, subfolder.name)
                corpus['ottoman'].append(ottoman_analysis)
                corpus['turkish'].append(turkish_analysis)
                corpus['parallel'].append((ottoman_analysis, turkish_analysis))
    
    return corpus

def analyze_parallel_texts(parallel_texts):
    ottoman_lengths = []
    turkish_lengths = []
    length_ratios = []

    for ottoman, turkish in parallel_texts:
        ottoman_words = ottoman['token_count']
        turkish_words = turkish['token_count']
        ottoman_lengths.append(ottoman_words)
        turkish_lengths.append(turkish_words)
        length_ratios.append(turkish_words / ottoman_words if ottoman_words > 0 else 0)

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

# 1. Corpus Size Metrics
def corpus_size_metrics(documents):
    total_tokens = sum(doc['token_count'] for doc in documents)
    vocabulary = set(word for doc in documents for word in doc['text'].split())
    avg_sentence_length = np.mean([doc['avg_sentence_length'] for doc in documents])
    
    print(f"Total Tokens: {total_tokens}")
    print(f"Vocabulary Size: {len(vocabulary)}")
    print(f"Average Sentence Length: {avg_sentence_length:.2f}")

# 2. Token Frequency Distribution (Zipf's Law)
def plot_zipf_law(documents):
    all_words = [word for doc in documents for word in doc['text'].split()]
    word_counts = Counter(all_words)
    counts = np.array(list(word_counts.values()))
    ranks = np.arange(1, len(counts)+1)
    indices = np.argsort(-counts)
    frequencies = counts[indices]

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Distribution (Zipf\'s Law)')
    plt.savefig(output_dir / 'zipf_law.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. N-gram Analysis
def plot_top_ngrams(documents, n=2, top_k=20):
    texts = [doc['text'] for doc in documents]
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(texts)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    top_ngrams = sorted(zip(ngrams, counts), key=lambda x: x[1], reverse=True)[:top_k]
    
    plt.figure(figsize=(12, 6))
    plt.barh(*zip(*reversed(top_ngrams)))
    plt.xlabel('Frequency')
    plt.title(f'Top {top_k} {n}-grams')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f'top_{n}grams.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Vocabulary Growth Curve
def plot_vocabulary_growth(documents):
    vocab = set()
    growth = []
    for doc in documents:
        vocab.update(doc['text'].split())
        growth.append(len(vocab))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(growth)+1), growth)
    plt.xlabel('Number of Documents')
    plt.ylabel('Cumulative Vocabulary Size')
    plt.title('Vocabulary Growth Curve')
    plt.savefig(output_dir / 'vocab_growth.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Topic Modeling with LDA
def perform_topic_modeling(documents, n_topics=10, n_top_words=15):
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform([doc['text'] for doc in documents])
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_words[f'Topic {topic_idx+1}'] = top_features
    
    df = pd.DataFrame(topic_words)
    
    # Heatmap of word importance
    word_importance = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    top_words_set = set(df.values.flatten())
    top_importance_df = pd.DataFrame(word_importance.T, index=feature_names, columns=[f'Topic {i+1}' for i in range(n_topics)])
    top_importance_df = top_importance_df.loc[list(top_words_set)]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_importance_df, cmap='viridis')
    plt.title('Heatmap of Word Importance Across Topics')
    plt.xlabel('Topics')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
# 6. t-SNE Visualization
def visualize_tsne(documents):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform([doc['text'] for doc in documents])
    
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X.toarray())
    
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(X_embedded)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Document Topics')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_topics.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. UMAP Visualization with Word Embeddings
def visualize_umap_with_embeddings(documents):
    # Load pre-trained FastText model (you'll need to have this model file)
    model = FastText.load('path_to_fasttext_model')
    
    def get_document_vector(doc):
        vectors = [model.wv[word] for word in doc.split() if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    
    document_vectors = [get_document_vector(doc) for doc in documents]
    
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(document_vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
    plt.title('UMAP Visualization of Documents using Word Embeddings')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Combine Plots (example)
def combined_plots(documents):
    plt.figure(figsize=(20, 10))
    
    # Token Distribution
    plt.subplot(2, 2, 1)
    all_words = [word for doc in documents for word in doc.split()]
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(20)
    plt.bar(*zip(*most_common))
    plt.title('Top 20 Most Common Words')
    plt.xticks(rotation=45, ha='right')
    
    # Vocabulary Growth
    plt.subplot(2, 2, 2)
    vocab = set()
    growth = [len(set(doc.split()).union(vocab)) for doc in documents]
    plt.plot(range(1, len(growth)+1), growth)
    plt.title('Vocabulary Growth')
    plt.xlabel('Number of Documents')
    plt.ylabel('Vocabulary Size')
    
    # Average Sentence Length
    plt.subplot(2, 2, 3)
    avg_lengths = [np.mean([len(sent.split()) for sent in doc.split('.')]) for doc in documents]
    plt.hist(avg_lengths, bins=20)
    plt.title('Distribution of Average Sentence Lengths')
    plt.xlabel('Average Sentence Length')
    plt.ylabel('Frequency')
    
    # Document Length Distribution
    plt.subplot(2, 2, 4)
    doc_lengths = [len(doc.split()) for doc in documents]
    plt.hist(doc_lengths, bins=20)
    plt.title('Distribution of Document Lengths')
    plt.xlabel('Document Length (words)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_info_file(corpus, output_dir):
    with open(output_dir / 'info.txt', 'w', encoding='utf-8') as f:
        f.write("Corpus Analysis Information\n")
        f.write("===========================\n\n")
        
        total_tokens = 0
        
        for corpus_type in ['ottoman', 'turkish']:
            f.write(f"{corpus_type.capitalize()} Corpus:\n")
            f.write(f"Number of documents: {len(corpus[corpus_type])}\n")
            tokens = sum(doc['token_count'] for doc in corpus[corpus_type])
            total_tokens += tokens
            f.write(f"Total tokens: {tokens}\n")
            vocab = set(word for doc in corpus[corpus_type] for word in doc['text'].split())
            f.write(f"Vocabulary size: {len(vocab)}\n")
            avg_sent_length = np.mean([doc['avg_sentence_length'] for doc in corpus[corpus_type]])
            f.write(f"Average sentence length: {avg_sent_length:.2f}\n\n")
            
            f.write("Individual file analysis:\n")
            for doc in corpus[corpus_type]:
                f.write(f"  {doc['file_name']}:\n")
                f.write(f"    Timestamp: {doc['timestamp']}\n")
                f.write(f"    Token count: {doc['token_count']}\n")
                f.write(f"    Unique tokens: {doc['unique_tokens']}\n")
                f.write(f"    Sentence count: {doc['sentence_count']}\n")
                f.write(f"    Avg sentence length: {doc['avg_sentence_length']:.2f}\n\n")
        
        f.write("Parallel Corpus:\n")
        f.write(f"Number of document pairs: {len(corpus['parallel'])}\n")
        ottoman_lengths = [len(ottoman['text'].split()) for ottoman, _ in corpus['parallel']]
        turkish_lengths = [len(turkish['text'].split()) for _, turkish in corpus['parallel']]
        f.write(f"Average Ottoman document length: {np.mean(ottoman_lengths):.2f} words\n")
        f.write(f"Average Turkish document length: {np.mean(turkish_lengths):.2f} words\n")
        length_ratios = [len(turkish['text'].split()) / len(ottoman['text'].split()) if len(ottoman['text'].split()) > 0 else 0 for ottoman, turkish in corpus['parallel']]
        f.write(f"Average Turkish/Ottoman length ratio: {np.mean(length_ratios):.2f}\n\n")
        
        f.write(f"Total tokens in the entire corpus: {total_tokens}\n")

    print(f"Info file generated: {output_dir / 'info.txt'}")

def analyze_text_file(file_path, folder_name=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = text.split()
    sentences = text.split('.')
    
    return {
        'file_name': folder_name if folder_name else file_path.name,
        'timestamp': datetime.datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        'text': text,
        'token_count': len(tokens),
        'unique_tokens': len(set(tokens)),
        'sentence_count': len(sentences),
        'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences])
    }

def analyze_sozluk(sozluk_path):
    with open(sozluk_path, 'r', encoding='utf-8') as f:
        sozluk = f.read().split('\n')
    
    word_pairs = [line.split('\t') for line in sozluk if '\t' in line]
    ottoman_words = [pair[0] for pair in word_pairs]
    turkish_words = [pair[1] for pair in word_pairs]
    
    return {
        'total_entries': len(word_pairs),
        'unique_ottoman_words': len(set(ottoman_words)),
        'unique_turkish_words': len(set(turkish_words)),
        'avg_ottoman_word_length': np.mean([len(word) for word in ottoman_words]),
        'avg_turkish_word_length': np.mean([len(word) for word in turkish_words])
    }

# Main execution
if __name__ == "__main__":
    # Load your corpus
    corpus = load_corpus()

    # Create output directory
    output_dir = root_dir / "corpus-texts" / "analysis_output"
    output_dir.mkdir(exist_ok=True)

    # Run analyses
    corpus_size_metrics(corpus['ottoman'])
    
    try:
        plot_zipf_law(corpus['ottoman'])
    except Exception as e:
        print(f"Error in Zipf's law plot: {e}")
    
    try:
        plot_top_ngrams(corpus['ottoman'], n=2)
        plot_top_ngrams(corpus['ottoman'], n=3)
    except Exception as e:
        print(f"Error in n-gram analysis: {e}")
    
    try:
        plot_vocabulary_growth(corpus['ottoman'])
    except Exception as e:
        print(f"Error in vocabulary growth plot: {e}")
    
    try:
        perform_topic_modeling(corpus['ottoman'])
    except Exception as e:
        print(f"Error in topic modeling: {e}")
    
    visualize_tsne(corpus['ottoman'])
    
    try:
        visualize_umap_with_embeddings(corpus['ottoman'])
    except Exception as e:
        print(f"Error in UMAP visualization: {e}")
    
    try:
        combined_plots(corpus['ottoman'])
    except Exception as e:
        print(f"Error in combined plots: {e}")

    # Parallel text analysis
    if corpus['parallel']:
        try:
            analyze_parallel_texts(corpus['parallel'])
        except Exception as e:
            print(f"Error in parallel text analysis: {e}")

    generate_info_file(corpus, output_dir)

    sozluk_path = root_dir / "corpus-texts" / "texts" / "sözlük.txt"
    if sozluk_path.exists():
        sozluk_analysis = analyze_sozluk(sozluk_path)
        with open(output_dir / 'info.txt', 'a', encoding='utf-8') as f:
            f.write("\nSözlük Analysis:\n")
            for key, value in sozluk_analysis.items():
                f.write(f"{key}: {value}\n")

    print(f"All analyses and visualizations complete. Check the output directory: {output_dir}")

