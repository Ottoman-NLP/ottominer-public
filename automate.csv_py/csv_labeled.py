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

def normalize_text(text):
    """Normalizes text by replacing common character variations and trimming spaces."""
    text = text.lower()
    text = re.sub(r'[\u2018\u2019\u201C\u201D]', '"', text)  # Replace curly quotes with standard quotes
    text = re.sub(r'[\u2013\u2014]', '-', text)  # Replace dashes
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    text = text.replace("â", "a").replace("î", "i").replace("û", "u")
    return text.strip()

def split_into_sentences(text):
    """Splits text into sentences based on punctuation and other rules, avoiding incorrect splits at ordinals and abbreviations."""
    text = re.sub(r'(?<=\w)-\s*\n\s*', '', text)  # Joins split words with hyphens at line breaks
    text = re.sub(r'\n', ' ', text)  # Replaces line breaks within sentences with a space

    # Prevent splitting after Roman numerals and single-letter abbreviations
    text = re.sub(r'\b([IVXLCDM]+\.)\s+', r'\1', text)  # Keep Roman numerals joined
    text = re.sub(r'\b([A-Z]\.)\s+', r'\1', text)  # Keep single-letter abbreviations joined

    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])\n|(?<=[.!?])(?=[A-Z])', text)
    
    sentences = [sub_sentence for sentence in sentences for sub_sentence in re.split(r'(?<=\S[;:])\s+', sentence)]
    
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def align_sentences(noisy_sentences, clean_sentences, similarity_threshold=0.3, position_weight=0.3):
    """Align sentences using a combination of similarity and position."""
    vectorizer = TfidfVectorizer()
    noisy_vectors = vectorizer.fit_transform(noisy_sentences)
    clean_vectors = vectorizer.transform(clean_sentences)

    aligned_pairs = []
    used_clean_indices = set()

    for i, noisy_sentence in enumerate(tqdm(noisy_sentences, desc="Aligning sentences")):
        similarities = cosine_similarity(noisy_vectors[i], clean_vectors)[0]
        position_scores = 1 - np.abs(np.arange(len(clean_sentences)) - i) / len(clean_sentences)
        combined_scores = (1 - position_weight) * similarities + position_weight * position_scores

        for j in np.argsort(combined_scores)[::-1]:
            if j not in used_clean_indices and combined_scores[j] >= similarity_threshold:
                aligned_pairs.append((noisy_sentence, clean_sentences[j]))
                used_clean_indices.add(j)
                break

    return aligned_pairs

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

noisy_text = read_file(original_text)
clean_text = read_file(cleaned_text)

if not noisy_text.strip() or not clean_text.strip():
    print("Error: One or both of the input files are empty.")
else:
    noisy_sentences = split_into_sentences(noisy_text)
    clean_sentences = split_into_sentences(clean_text)

    aligned_pairs = align_sentences(noisy_sentences, clean_sentences)
    aligned_noisy, aligned_clean = zip(*aligned_pairs) if aligned_pairs else ([], [])

    if not aligned_noisy or not aligned_clean:
        print("No aligned sentences found.")
    else:
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

visualize_alignment_quality(aligned_noisy, aligned_clean)
print(f"Alignment quality visualization saved at {sp_ / 'alignment_quality.png'}")

print("Script execution completed.")