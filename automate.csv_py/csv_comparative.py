import os
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

bd_ = Path(__file__).resolve().parents[2]
sp_ = bd_ / 'corpus-texts' / 'automated.csv.results'
id_ = bd_ / 'corpus-texts' / 'automated.csv'

os.makedirs(sp_, exist_ok=True)

aligned_noisy_texts = []
aligned_clean_texts = []
misaligned_noisy = []
misaligned_clean = []

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

def align_sentences_by_similarity(noisy_sentences, clean_sentences, similarity_threshold=0.5):
    """Align sentences using cosine similarity of normalized TF-IDF vectors, with a threshold to filter low similarities."""
    if not noisy_sentences or not clean_sentences:
        print("One of the sentence lists is empty. Skipping this pair.")
        return [], []

    normalized_noisy = [normalize_text(sent) for sent in noisy_sentences]
    normalized_clean = [normalize_text(sent) for sent in clean_sentences]
    
    vectorizer = TfidfVectorizer().fit(normalized_noisy + normalized_clean)
    noisy_vectors = vectorizer.transform(normalized_noisy)
    clean_vectors = vectorizer.transform(normalized_clean)
    
    similarity_matrix = cosine_similarity(noisy_vectors, clean_vectors)
    aligned_noisy = []
    aligned_clean = []

    for i in range(len(noisy_sentences)):
        max_sim_index = np.argmax(similarity_matrix[i])
        max_similarity = similarity_matrix[i, max_sim_index]

        if max_similarity >= similarity_threshold:
            aligned_noisy.append(noisy_sentences[i])
            aligned_clean.append(clean_sentences[max_sim_index])
        else:
            misaligned_noisy.append(noisy_sentences[i])
            misaligned_clean.append(clean_sentences[max_sim_index])

        similarity_matrix[:, max_sim_index] = -1

    return aligned_noisy, aligned_clean

extracted_files = {file.stem.replace('_extract', ''): file for file in id_.glob('*_extract.txt')}
cleaned_files = {file.stem.replace('_clean', ''): file for file in id_.glob('*_clean.txt')}

print(f"Extracted files found: {list(extracted_files.keys())}")
print(f"Cleaned files found: {list(cleaned_files.keys())}")

for base_name in sorted(extracted_files.keys() & cleaned_files.keys()):
    extract_path = extracted_files[base_name]
    clean_path = cleaned_files[base_name]
    
    print(f"Processing: {base_name}")

    noisy_text = read_file(extract_path)
    clean_text = read_file(clean_path)
    
    if not noisy_text.strip() or not clean_text.strip():
        print(f"Skipping empty files: {base_name}")
        continue
    
    noisy_sentences = split_into_sentences(noisy_text)
    clean_sentences = split_into_sentences(clean_text)

    aligned_noisy, aligned_clean = align_sentences_by_similarity(noisy_sentences, clean_sentences)

    if not aligned_noisy or not aligned_clean:
        print(f"No aligned sentences for: {base_name}")
        continue

    aligned_noisy_texts.extend(aligned_noisy)
    aligned_clean_texts.extend(aligned_clean)

    data = {'noisy': aligned_noisy, 'clean': aligned_clean}
    df = pd.DataFrame(data)
    output_csv_path = sp_ / f'{base_name}_aligned.csv'
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    num_tokens_noisy = sum(len(sentence.split()) for sentence in aligned_noisy)
    num_tokens_clean = sum(len(sentence.split()) for sentence in aligned_clean)
    metadata.append({
        'file_name': base_name,
        'num_noisy_sentences': len(aligned_noisy),
        'num_clean_sentences': len(aligned_clean),
        'num_noisy_tokens': num_tokens_noisy,
        'num_clean_tokens': num_tokens_clean
    })

all_data = {'noisy': aligned_noisy_texts, 'clean': aligned_clean_texts}
all_df = pd.DataFrame(all_data)
all_aligned_csv_path = sp_ / 'all_aligned_sentences.csv'
all_df.to_csv(all_aligned_csv_path, index=False, encoding='utf-8')

metadata_df = pd.DataFrame(metadata)

total_noisy_tokens = sum(md['num_noisy_tokens'] for md in metadata)
total_clean_tokens = sum(md['num_clean_tokens'] for md in metadata)
total_metadata = pd.DataFrame([{
    'file_name': 'TOTAL',
    'num_noisy_sentences': sum(md['num_noisy_sentences'] for md in metadata),
    'num_clean_sentences': sum(md['num_clean_sentences'] for md in metadata),
    'num_noisy_tokens': total_noisy_tokens,
    'num_clean_tokens': total_clean_tokens
}])

metadata_df = pd.concat([metadata_df, total_metadata], ignore_index=True)

metadata_csv_path = sp_ / 'metadata_summary.csv'
metadata_df.to_csv(metadata_csv_path, index=False, encoding='utf-8')

misaligned_data = {'noisy': misaligned_noisy, 'clean': misaligned_clean}
misaligned_df = pd.DataFrame(misaligned_data)
misaligned_csv_path = sp_ / 'misaligned_sentences.csv'
misaligned_df.to_csv(misaligned_csv_path, index=False, encoding='utf-8')

print(f"All aligned sentences saved at {all_aligned_csv_path}")
print(f"Metadata summary saved at {metadata_csv_path}")
print(f"Misaligned sentences saved at {misaligned_csv_path}")