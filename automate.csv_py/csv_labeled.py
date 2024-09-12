import os
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import time
import gc
from datasketch import MinHash, MinHashLSH
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

NUM_PERM = 128

bd_ = Path(__file__).resolve().parents[2]
_sp_ = bd_ / 'corpus-texts'
sp_ = bd_ / 'corpus-texts' / 'datasets' # path for files to be saved


original_text = _sp_ / 'datasets' / 'noisy_corpus.txt'
cleaned_text = _sp_ / 'datasets' /'clean_corpus.txt'
os.makedirs(sp_, exist_ok=True)

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def split_into_paragraphs(text):
    return re.split(r'\n\s*\n', text)

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def create_minhash(text, num_perm=NUM_PERM):
    minhash = MinHash(num_perm=num_perm)
    for ngram in get_ngrams(text, 3):
        minhash.update(ngram.encode('utf-8'))
    return minhash

def get_ngrams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def batch_align_sentences(noisy_batch, clean_sentences, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for i, clean_sentence in enumerate(clean_sentences):
        lsh.insert(f"clean_{i}", create_minhash(clean_sentence))
    
    aligned_pairs = []
    for noisy_sentence in noisy_batch:
        query_minhash = create_minhash(noisy_sentence)
        matches = lsh.query(query_minhash)
        if matches:
            best_match = max(matches, key=lambda x: int(x.split('_')[1]))
            aligned_pairs.append((noisy_sentence, clean_sentences[int(best_match.split('_')[1])]))
    
    return aligned_pairs

def process_batch(args):
    noisy_batch, clean_sentences, threshold = args
    return batch_align_sentences(noisy_batch, clean_sentences, threshold)

def distance(l1, l2):
    mean = (l1 + l2) / 2
    variance = 6.8 if mean < 8 else 1.3 * mean
    z = (l1 - l2) / math.sqrt(2 * variance)
    return -math.log(math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))

def gale_church_align(source_sentences, target_sentences):
    def sentence_length(sentence):
        return len(sentence.split())

    original_text = _sp_ / 'datasets' / 'noisy_corpus.txt'


    m, n = len(source_sentences), len(target_sentences)
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = distance(sentence_length(source_sentences[i-1]), sentence_length(target_sentences[j-1]))
            dp[i][j] = min(dp[i][j], dp[i-1][j-1] + cost)

    aligned_pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + distance(sentence_length(source_sentences[i-1]), sentence_length(target_sentences[j-1])):
            aligned_pairs.append((source_sentences[i-1], target_sentences[j-1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j]:
            i -= 1
        else:
            j -= 1

    return list(reversed(aligned_pairs))

def align_paragraphs(noisy_text, clean_text):
    noisy_paragraphs = split_into_paragraphs(noisy_text)
    clean_paragraphs = split_into_paragraphs(clean_text)
    
    aligned_pairs = []
    for noisy_para, clean_para in zip(noisy_paragraphs, clean_paragraphs):
        noisy_sentences = split_into_sentences(noisy_para)
        clean_sentences = split_into_sentences(clean_para)
        
        noisy_sentences = [preprocess_text(s) for s in noisy_sentences]
        clean_sentences = [preprocess_text(s) for s in clean_sentences]
        
        alignment = gale_church_align(noisy_sentences, clean_sentences)
        aligned_pairs.extend(alignment)
    
    return aligned_pairs

def post_process(aligned_pairs):
    return [(noisy, clean) for noisy, clean in aligned_pairs 
            if 0.5 <= len(noisy) / len(clean) <= 2]

def main():
    print("Reading input files...")
    noisy_text = read_file(original_text)
    clean_text = read_file(cleaned_text)

    if not noisy_text.strip() or not clean_text.strip():
        print("Error: One or both of the input files are empty.")
        return

    print("Aligning paragraphs and sentences...")
    aligned_pairs = align_paragraphs(noisy_text, clean_text)

    if not aligned_pairs:
        print("No aligned sentences found.")
        return

    print("Post-processing...")
    aligned_pairs = post_process(aligned_pairs)

    print("Saving results...")
    aligned_noisy, aligned_clean = zip(*aligned_pairs)
    data = {'noisy': aligned_noisy, 'clean': aligned_clean}
    df = pd.DataFrame(data)
    output_csv_path = sp_ / 'aligned_sentences.csv'
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    num_tokens_noisy = sum(len(sentence.split()) for sentence in aligned_noisy)
    num_tokens_clean = sum(len(sentence.split()) for sentence in aligned_clean)
    metadata = [{
        'file_name': 'merged_clean_and_final_clean',
        'num_noisy_sentences': len(aligned_noisy),
        'num_clean_sentences': len(aligned_clean),
        'num_noisy_tokens': num_tokens_noisy,
        'num_clean_tokens': num_tokens_clean
    }]

    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = sp_ / 'metadata_summary.csv'
    metadata_df.to_csv(metadata_csv_path, index=False, encoding='utf-8')

    print(f"Aligned sentences saved at {output_csv_path}")
    print(f"Metadata summary saved at {metadata_csv_path}")
    print("Script execution completed.")

if __name__ == '__main__':
    main()