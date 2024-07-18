import os
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import pandas as pd
import sys
import ssl
import re
import time
from nltk.stem import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from networkx.algorithms import community
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cupy as cp
import dask.array as da
from dask.distributed import Client, LocalCluster

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../etc/anim')))
from progress import ProgressBar

# Due to NLTK SSL certificate issue!
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

stemmer = PorterStemmer()

def read_text_files(directory_or_file, progress_bar=None):
    texts = []
    if os.path.isdir(directory_or_file):
        filenames = [f for f in os.listdir(directory_or_file) if f.endswith(".txt")]
        total_files = len(filenames)
        for i, filename in enumerate(filenames):
            with open(os.path.join(directory_or_file, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
            if progress_bar:
                progress_bar.update("Reading Text Files", (i + 1) / total_files * 100, "Next File", total_files)
    else:
        with open(directory_or_file, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def log_graph_metrics(G, output_dir, progress_bar=None):
    if len(G) == 0:
        logging.warning("Graph is empty, skipping metric logging.")
        return

    try:
        metrics = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G),
            'density': nx.density(G),
            'number_of_communities': len(list(nx.community.greedy_modularity_communities(G)))
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, 'graph_metrics.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            for key, value in metrics.items():
                file.write(f'{key}: {value}\n')

        if progress_bar:
            progress_bar.update("Logging Graph Metrics", 100, "Completed")
                
        logging.info(f"Graph metrics logged to {file_path}")
    except Exception as e:
        logging.error(f"Error logging graph metrics: {e}")




def visualize_graph(G, output_dir, max_nodes=1000, progress_bar=None):
    if len(G) == 0:
        logging.warning("Graph is empty, skipping visualization.")
        return
    
    try:
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()

        logging.info(f"Starting graph visualization with {total_nodes} nodes and {total_edges} edges.")
        start_time = time.time()

        plt.figure(figsize=(10, 10))

        if total_nodes > max_nodes:
            logging.info(f"Graph is too large ({total_nodes} nodes). Visualizing a subgraph of {max_nodes} nodes.")
            G = G.subgraph(list(G.nodes)[:max_nodes])

        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=10 if total_nodes > max_nodes else 50, font_size=8 if total_nodes > max_nodes else 10)

        plt.savefig(os.path.join(output_dir, 'network_graph.png'))
        plt.close()

        end_time = time.time()
        logging.info(f"Graph visualization completed in {end_time - start_time:.2f} seconds.")
        
        if progress_bar:
            progress_bar.update("Visualizing Graph", 100, "Completed")
    except MemoryError as me:
        logging.error(f"Memory error during graph visualization: {me}. Try visualizing a smaller subgraph.")
    except Exception as e:
        logging.error(f"Error visualizing graph: {e}")




def generate_frequency_lists(texts):
    all_tokens = []
    for text in texts:
        tokens = tokenize_and_clean(text)
        all_tokens.extend(tokens)
    
    frequency_counts = Counter(all_tokens)
    rank_order_list = frequency_counts.most_common()
    
    return rank_order_list, all_tokens


def normalize_frequencies(frequency_counts, total_tokens):
    return {word: count/total_tokens for word, count in frequency_counts.items()}


def generate_concordances(texts, search_terms, window=5):
    concordances = defaultdict(list)
    for term in search_terms:
        for text in texts:
            tokens = word_tokenize(text)
            text_length = len(tokens)
            for i, token in enumerate(tokens):
                if token == term:
                    start = max(i - window, 0)
                    end = min(i + window + 1, text_length)
                    concordances[term].append(' '.join(tokens[start:end]))
    return concordances



def generate_lemma_frequency_lists(texts):
    lemma_tokens = []

    for text in texts:
        tokens = tokenize_and_clean(text)
        lemmatized_tokens = [stemmer.stem(token) for token in tokens]
        lemma_tokens.extend(lemmatized_tokens)
    
    lemma_frequency_counts = Counter(lemma_tokens)
    lemma_rank_order_list = lemma_frequency_counts.most_common()
    
    return lemma_rank_order_list, lemma_tokens



def save_frequency_list(rank_order, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    rank_order_df = pd.DataFrame(rank_order, columns=['Token', 'Frequency'])
    rank_order_df.to_csv(file_path, index=False)


def compare_with_heards_law(original_frequency, lemmatized_frequency):
    original_total = sum(freq for _, freq in original_frequency)
    lemmatized_total = sum(freq for _, freq in lemmatized_frequency)
    
    heard_ratio = lemmatized_total / original_total
    accuracy = heard_ratio * 100
    
    return heard_ratio, accuracy


def compare_with_heaps_law(original_frequency, total_tokens, k=10, beta=0.7):
    vocabulary_size = len(original_frequency)
    estimated_vocabulary_size = k * (total_tokens ** beta)
    
    heaps_ratio = vocabulary_size / estimated_vocabulary_size
    accuracy = heaps_ratio * 100  
    
    return heaps_ratio, accuracy


def log_metrics(metrics, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        for key, value in metrics.items():
            file.write(f'{key}: {value}\n')
            
def detect_noise(texts):
    noise_patterns = [
        r'[^\w\s,.?!]',  # non-alphanumeric
        r'^\d+$',        # standalone digits
        r'[^\x00-\x7F]+' # non-ASCII
    ]
    noise_counts = {pattern: 0 for pattern in noise_patterns}
    for sentence in texts:
        for pattern in noise_patterns:
            if re.search(pattern, sentence):
                noise_counts[pattern] += 1
    return noise_counts


def tfidf_analysis(texts, output_dir):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    
    # Save TF-IDF scores to a CSV file
    df.to_csv(os.path.join(output_dir, 'tfidf_scores.csv'), index=False)
    
    # Create and save a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap='viridis')
    plt.title('TF-IDF Heatmap')
    plt.savefig(os.path.join(output_dir, 'tfidf_heatmap.png'))
    plt.close()

def tokenize_and_clean(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if token.isalnum()]
    return cleaned_tokens

def generate_network_graph(texts, window=5, progress_bar=None, output_path="network_graph.gexf"):
    cluster = LocalCluster()
    client = Client(cluster)
    try:
        G = nx.Graph()
        total_texts = len(texts)
        processed_texts = 0

        def process_tokens(tokens):
            token_length = len(tokens)
            edges = []
            for i in range(token_length):
                for j in range(max(0, i - window), min(token_length, i + window + 1)):
                    if i != j:
                        edges.append((tokens[i], tokens[j]))
            return edges

        scattered_texts = client.scatter(texts, broadcast=True)

        futures = client.map(tokenize_and_clean, scattered_texts, pure=False)
        for i, future in enumerate(futures):
            try:
                tokens = future.result(timeout=60)
                edges = process_tokens(tokens)
                G.add_edges_from(edges)
                processed_texts += 1

                if progress_bar:
                    progress_bar.update("Generating Network Graph", (processed_texts / total_texts) * 100, "Next Text")
                else:
                    logging.info(f"Processed {processed_texts}/{total_texts} texts")
            except Exception as e:
                logging.error(f"Error processing text {i}: {e}")

        nx.write_gexf(G, output_path)
        logging.info(f"Network graph saved to {output_path}")
    finally:
        client.close()
        cluster.close()

    return G


def analyze_corpus(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pb = ProgressBar()
    start_time = time.time()

    # Reading texts
    t = pb.start(f"Reading texts from {input_dir}", "Reading Texts", "detailed", input_dir)
    logging.info(f"Reading texts from {input_dir}")
    texts = read_text_files(input_dir, progress_bar=pb)
    read_duration = time.time() - start_time
    pb.stop(t)
    logging.info(f"Reading texts completed in {read_duration:.2f} seconds.")

    # Noise detection
    noise_counts = detect_noise(texts)

    # Generate network graph
    t = pb.start("Generating Network Graph", "Network Graph Generation", "detailed", output_dir)
    logging.info("Generating Network Graph")
    G = generate_network_graph(texts, progress_bar=pb, output_path=os.path.join(output_dir, "network_graph.gexf"))
    pb.stop(t)

    if os.path.exists('../../var/results/corpus_analytics/network_graph.gexf'):
        t = pb.start("Generating Network Graph Image", "Network Graph Generation", "detailed", '../../var/results/corpus_analytics')
        logging.info("Generating Network Graph Image")
        G = nx.read_gexf('../../var/results/corpus_analytics/network_graph.gexf')
        visualize_graph(G, output_dir, progress_bar=pb)
        pb.stop(t)
    else:
        logging.warning("Generated graph is empty. Skipping metrics and visualization.")

    t = pb.start("Generating Frequency Lists", "Frequency List Generation", "detailed", output_dir)
    logging.info("Generating Frequency Lists")
    rank_order, all_tokens = generate_frequency_lists(texts)
    normalized_rank_order = normalize_frequencies(dict(rank_order), len(all_tokens))
    normalized_rank_order_list = sorted(normalized_rank_order.items(), key=lambda item: item[1], reverse=True)
    pb.update("Generating Frequency Lists", 50, "Saving Frequency Lists", 100)
    save_frequency_list(rank_order, 'rank_order_frequency_list.csv', output_dir)
    save_frequency_list(normalized_rank_order_list, 'normalized_rank_order_frequency_list.csv', output_dir)
    pb.stop(t)
    
    t = pb.start("Generating Lemma Frequency Lists", "Lemma Frequency List Generation", "detailed", output_dir)
    logging.info("Generating Lemma Frequency Lists")
    lemma_rank_order, lemma_tokens = generate_lemma_frequency_lists(texts)
    pb.update("Generating Lemma Frequency Lists", 50, "Saving Lemma Frequency Lists", 100)
    save_frequency_list(lemma_rank_order, 'lemma_rank_order_frequency_list.csv', output_dir)
    pb.stop(t)

    heard_ratio, accuracy = compare_with_heards_law(rank_order, lemma_rank_order)
    total_tokens = len(all_tokens)
    heaps_ratio, heaps_accuracy = compare_with_heaps_law(rank_order, total_tokens)

    search_terms = [word for word, count in rank_order][:10]
    t = pb.start("Generating Concordances", "Concordance Generation", "detailed", output_dir)
    logging.info("Generating Concordances")
    concordances = generate_concordances(texts, search_terms)
    pb.update("Generating Concordances", 50, "Concordance Generation", 100)
    pb.stop(t)

    t = pb.start("Performing TF-IDF Analysis", "TF-IDF Analysis", "detailed", output_dir)
    logging.info("Performing TF-IDF Analysis")
    tfidf_analysis(texts, output_dir)
    pb.update("Performing TF-IDF Analysis", 50, "Saving TF-IDF Results", 100)
    pb.stop(t)

    unique_tokens = len(set(all_tokens))
    token_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
    avg_sentence_length = sum(len(word_tokenize(sent)) for sent in texts) / len(texts) if len(texts) > 0 else 0
    std_dev_sentence_length = (sum((len(word_tokenize(sent)) - avg_sentence_length) ** 2 for sent in texts) / len(texts)) ** 0.5 if len(texts) > 0 else 0

    metrics = {
        'Total Tokens': total_tokens,
        'Unique Tokens': unique_tokens,
        'Token Diversity': token_diversity,
        'Average Sentence Length': avg_sentence_length,
        'Standard Deviation of Sentence Length': std_dev_sentence_length,
        'Heard\'s Law Ratio': heard_ratio,
        'Approximate Accuracy': accuracy,
        'Heap\'s Law Ratio': heaps_ratio,
        'Heap\'s Law Accuracy': heaps_accuracy,
        'Read Duration (seconds)': read_duration
    }

    for pattern, count in noise_counts.items():
        metrics[f'Noise Pattern {pattern}'] = count

    log_metrics(metrics, 'analytics_summary.txt', output_dir)

    return metrics

def compare_metrics(metrics1, metrics2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'comparative_analysis.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        for key in metrics1:
            file.write(f'{key}:\n')
            file.write(f'  Original: {metrics1[key]}\n')
            file.write(f'  Cleaned: {metrics2[key]}\n')
            file.write(f'  Difference: {metrics2[key] - metrics1[key]}\n')
            file.write('\n')

    noise_keys = [key for key in metrics1.keys() if 'Noise Pattern' in key]
    noise_data_original = [metrics1[key] for key in noise_keys]
    noise_data_cleaned = [metrics2[key] for key in noise_keys]

    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(noise_keys))

    bars1 = ax.bar(index, noise_data_original, bar_width, label='Original')
    bars2 = ax.bar(index + bar_width, noise_data_cleaned, bar_width, label='Cleaned')

    ax.set_xlabel('Noise Patterns')
    ax.set_ylabel('Counts')
    ax.set_title('Noise Pattern Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(noise_keys)
    ax.legend()

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'noise_comparison.png'))
    plt.close(fig)

def main():
    input_dir_original = '../../LLM/texture/txts'
    input_dir_cleaned = '../../cleaned_data'
    output_dir_original = '../../var/results/corpus_analytics'
    output_dir_cleaned = '../../var/results/clean_corpus_analytics'
    output_dir_comparative = '../../var/results/comparative_corpus'
    

    metrics_original = analyze_corpus(input_dir_original, output_dir_original)
    metrics_cleaned = analyze_corpus(input_dir_cleaned, output_dir_cleaned)

    compare_metrics(metrics_original, metrics_cleaned, output_dir_comparative)

if __name__ == "__main__":
    main()