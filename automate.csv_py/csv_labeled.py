import os
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, vstack, csr_matrix
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from difflib import SequenceMatcher
import time
import gc

bd_ = Path(__file__).resolve().parents[2]
_sp_ = bd_ / 'corpus-texts'
sp_ = bd_ / 'corpus-texts' / 'automated.csv.results'
id_ = bd_ / 'corpus-texts' / 'automated.csv'

original_text = _sp_ / 'merged_clean.txt'
cleaned_text = _sp_ / 'final_clean.txt'
os.makedirs(sp_, exist_ok=True)

aligned_noisy_texts = []
aligned_clean_texts = []

metadata = []

def read_file(file_path):
    """Reads and returns the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def normalize_text(text, is_noisy=True):
    if is_noisy:
        return text.strip()
    else:

        text = text.lower()
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()

def split_into_sentences(text, is_noisy=True):
    if is_noisy:

        return re.split(r'(?<=[.!?])\s+', text)
    else:

        return split_into_sentences(text)

def align_sentences_chunk(chunk, clean_sentences, vectorizer, similarity_threshold, position_weight):
    start_time = time.time()
    chunk_vectors = vectorizer.transform(chunk)
    clean_vectors = vectorizer.transform(clean_sentences)
    aligned_pairs = []
    for i, (noisy_sentence, noisy_vector) in enumerate(zip(chunk, chunk_vectors)):
        if i % 100 == 0:
            print(f"Processing sentence {i}/{len(chunk)} in chunk")
        similarities = cosine_similarity(noisy_vector, clean_vectors)[0]
        position_scores = 1 - np.abs(np.arange(len(clean_sentences)) - i) / len(clean_sentences)
        combined_scores = (1 - position_weight) * similarities + position_weight * position_scores
        best_match = np.argmax(combined_scores)
        if combined_scores[best_match] >= similarity_threshold:
            aligned_pairs.append((noisy_sentence, clean_sentences[best_match]))
    print(f"Chunk processed in {time.time() - start_time:.2f} seconds")
    return aligned_pairs

def align_sentences(noisy_sentences, clean_sentences, similarity_threshold=0.3, position_weight=0.3):
    print("Preparing for sentence alignment...")
    vectorizer = TfidfVectorizer(lowercase=False)
    vectorizer.fit(tqdm(noisy_sentences + clean_sentences, desc="Fitting TF-IDF vectorizer"))
    
    chunk_size = max(1, len(noisy_sentences) // multiprocessing.cpu_count())
    chunks = [noisy_sentences[i:i+chunk_size] for i in range(0, len(noisy_sentences), chunk_size)]
    
    print(f"Aligning sentences using {len(chunks)} chunks...")
    with multiprocessing.Pool() as pool:
        aligned_pairs = []
        for result in tqdm(pool.imap_unordered(partial(align_sentences_chunk, 
                                                       clean_sentences=clean_sentences,
                                                       vectorizer=vectorizer,
                                                       similarity_threshold=similarity_threshold,
                                                       position_weight=position_weight), 
                                               chunks),
                           total=len(chunks),
                           desc="Aligning chunks"):
            aligned_pairs.extend(result)
    
    return aligned_pairs

def memory_efficient_global_alignment(noisy_sentences, clean_sentences):
    print("Performing memory-efficient global alignment...")
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    m, n = len(noisy_sentences), len(clean_sentences)
    current_row = [0] * (n + 1)
    
    aligned_pairs = []
    for i in tqdm(range(1, m + 1), desc="Global alignment"):
        previous_row, current_row = current_row, [0] * (n + 1)
        for j in range(1, n + 1):
            match = previous_row[j-1] + similarity(noisy_sentences[i-1], clean_sentences[j-1])
            current_row[j] = max(previous_row[j], current_row[j-1], match)
        
        # Backtrack for this row
        if i % 1000 == 0 or i == m:
            j = n
            while j > 0 and i > 0:
                if current_row[j] == previous_row[j-1] + similarity(noisy_sentences[i-1], clean_sentences[j-1]):
                    aligned_pairs.append((noisy_sentences[i-1], clean_sentences[j-1]))
                    i -= 1
                    j -= 1
                elif current_row[j] == previous_row[j]:
                    i -= 1
                else:
                    j -= 1
            
            # Clear memory
            noisy_sentences = noisy_sentences[i:]
            m = len(noisy_sentences)
            i = 0
            gc.collect()
    
    return list(reversed(aligned_pairs))

def visualize_alignment_quality(aligned_noisy, aligned_clean):
    lengths_noisy = [len(sent.split()) for sent in aligned_noisy]
    lengths_clean = [len(sent.split()) for sent in aligned_clean]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths_noisy, lengths_clean, alpha=0.5)
    plt.xlabel('Noisy Sentence Length')
    plt.ylabel('Clean Sentence Length')
    plt.title('Alignment Quality: Sentence Lengths')
    plt.savefig(sp_ / 'alignment_quality.png')
    plt.close()

def main():
    print("Reading input files...")
    noisy_text = read_file(original_text)
    clean_text = read_file(cleaned_text)

    if not noisy_text.strip() or not clean_text.strip():
        print("Error: One or both of the input files are empty.")
    else:
        print("Splitting sentences...")
        noisy_sentences = split_into_sentences(noisy_text, is_noisy=True)
        clean_sentences = split_into_sentences(clean_text, is_noisy=False)

        print("Starting two-stage alignment process...")
        initial_aligned_pairs = align_sentences(noisy_sentences, clean_sentences)
        final_aligned_pairs = memory_efficient_global_alignment([pair[0] for pair in initial_aligned_pairs],
                                                               [pair[1] for pair in initial_aligned_pairs])

        aligned_noisy, aligned_clean = zip(*final_aligned_pairs) if final_aligned_pairs else ([], [])

        if not aligned_noisy or not aligned_clean:
            print("No aligned sentences found.")
        else:
            print("Saving results...")
            aligned_noisy_texts.extend(aligned_noisy)
            aligned_clean_texts.extend(aligned_clean)

            data = {'noisy': aligned_noisy, 'clean': aligned_clean}
            df = pd.DataFrame(data)
            output_csv_path = sp_ / 'aligned_sentences.csv'
            df.to_csv(output_csv_path, index=False, encoding='utf-8')

            num_tokens_noisy = sum(len(sentence.split()) for sentence in aligned_noisy)
            num_tokens_clean = sum(len(sentence.split()) for sentence in aligned_clean)
            metadata.append({
                'file_name': 'merged_clean_and_final_clean',
                'num_noisy_sentences': len(aligned_noisy),
                'num_clean_sentences': len(aligned_clean),
                'num_noisy_tokens': num_tokens_noisy,
                'num_clean_tokens': num_tokens_clean
            })

            metadata_df = pd.DataFrame(metadata)
            metadata_csv_path = sp_ / 'metadata_summary.csv'
            metadata_df.to_csv(metadata_csv_path, index=False, encoding='utf-8')

            print(f"Aligned sentences saved at {output_csv_path}")
            print(f"Metadata summary saved at {metadata_csv_path}")

    print("Generating alignment quality visualization...")
    visualize_alignment_quality(aligned_noisy, aligned_clean)
    print(f"Alignment quality visualization saved at {sp_ / 'alignment_quality.png'}")

    print("Script execution completed.")

if __name__ == '__main__':
    main()