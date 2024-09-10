from pathlib import Path
import sys
import os
import chardet
import re

bd_ = Path(__file__).resolve().parents[2]

input_dir = bd_ / "corpus-texts" / "clean_corpus_text"
output_file = input_dir / "merged_all.txt"
os.makedirs(input_dir, exist_ok=True)

def detect_encoding(file_path) -> str | None:
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_file(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def normalize_text(text):
    """Normalizes text by replacing common character variations and trimming spaces."""
    text = text.lower()
    text = re.sub(r'[\u2018\u2019\u201C\u201D]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    text = text.replace("â", "a").replace("î", "i").replace("û", "u")
    return text.strip()

def split_into_sentences(text):
    """Splits text into sentences based on punctuation and other rules, avoiding incorrect splits at ordinals and abbreviations."""
    text = re.sub(r'(?<=\w)-\s*\n\s*', '', text)
    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'\b([IVXLCDM]+\.)\s+', r'\1', text)
    text = re.sub(r'\b([A-Z]\.)\s+', r'\1', text)

    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])\n|(?<=[.!?])(?=[A-Z])', text)
    
    sentences = [sub_sentence for sentence in sentences for sub_sentence in re.split(r'(?<=\S[;:])\s+', sentence)]
    
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def merge_corpus_texts():
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for file in input_dir.glob("*.txt"):
            text = read_file(file)
            out_file.write(text)
            out_file.write('\n')

def clean_corpus_texts():
    with open(output_file, 'r', encoding='utf-8') as in_file:
        text = in_file.read()
    

    text = normalize_text(text.lower())
    text = re.sub(r'\d+', '', text)
    sentences = split_into_sentences(text)
    
    cleaned_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\b[a-z]\b', '', sentence)
        sentence = ' '.join(sentence.split())
        
        if sentence:
            cleaned_sentences.append(sentence)
    cleaned_text = ' '.join(cleaned_sentences)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(cleaned_text)


if __name__ == "__main__":
    merge_corpus_texts()
    clean_corpus_texts()
