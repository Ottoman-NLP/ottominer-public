from pathlib import Path
import re
import chardet
import os


bd_ = Path(__file__).resolve().parents[2]

input_dir = bd_ / "corpus-texts" / "txts"
output_file = input_dir / "datasets"
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

def normalize_ottoman_characters(text):
    char_map = {
        'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'İ',
        'û': 'u', 'Û': 'U',
        'ı': 'ı', 'I': 'İ',
        'ğ': 'ğ', 'Ğ': 'Ğ',
        'ü': 'ü', 'Ü': 'Ü',
        'ö': 'ö', 'Ö': 'Ö',
        'ş': 'ş', 'Ş': 'Ş',
        'ç': 'ç', 'Ç': 'Ç'
    }
    return ''.join(char_map.get(c, c) for c in text)

def split_into_sentences(text, clean=False):
    text = re.sub(r'(?<=\w)-\s*\n\s*', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\b([IVXLCDM]+\.)\s+', r'\1', text)
    text = re.sub(r'\b([A-Z]\.)\s+', r'\1', text)
    
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])\n|(?<=[.!?])(?=[A-Z])', text)
    sentences = [sub_sentence for sentence in sentences for sub_sentence in re.split(r'(?<=\S[;:])\s+', sentence)]
    
    if clean:
        sentences = [re.sub(r'\s+', ' ', s.strip()) for s in sentences]
        sentences = [normalize_ottoman_characters(s) for s in sentences]
        sentences = [s for s in sentences if len(s) > 10]  # Remove very short sentences
    
    return [s.strip() for s in sentences if s.strip()]

def process_files(input_dir, clean_output, noisy_output):
    clean_text, noisy_text = "", ""
    for file in Path(input_dir).glob('*.txt'):
        text = read_file(file)
        clean_sentences = split_into_sentences(text, clean=True)
        noisy_sentences = split_into_sentences(text, clean=False)
        
        clean_text += '\n'.join(clean_sentences) + '\n\n'
        noisy_text += '\n'.join(noisy_sentences) + '\n\n'
    
    with open(clean_output, 'w', encoding='utf-8') as f:
        f.write(clean_text.strip())
    with open(noisy_output, 'w', encoding='utf-8') as f:
        f.write(noisy_text.strip())

if __name__ == "__main__":
    input_dir = Path('corpus-texts/txts')
    clean_output = Path('corpus-texts/datasets/clean.txt')
    noisy_output = Path('corpus-texts/datasets/noisy.txt')
    process_files(input_dir, clean_output, noisy_output)
