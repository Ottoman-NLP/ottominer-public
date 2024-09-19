import json
from typing import List, Tuple
from collections import Counter
from pathlib import Path

goldset_dir = Path(__file__).parent.parent.parent / 'corpus-texts' / 'datasets' / 'goldset.json'

def load_data(file_path: str) -> List[Tuple[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item['noisy'], item['clean']) for item in data]

def character_tokenize(text: str) -> List[str]:
    # Add normalization rules here
    text = text.replace('â', 'a').replace('î', 'i').replace('û', 'u')
    return [char for char in text if char.strip()]

def analyze_data(data: List[Tuple[str, str]]):
    noisy_chars = Counter()
    clean_chars = Counter()
    
    for noisy, clean in data:
        noisy_chars.update(noisy)
        clean_chars.update(clean)
    
    print(f"Unique characters in noisy text: {len(noisy_chars)}")
    print(f"Unique characters in clean text: {len(clean_chars)}")
    print(f"Most common noisy characters: {noisy_chars.most_common(10)}")
    print(f"Most common clean characters: {clean_chars.most_common(10)}")

def preprocess_data(data: List[Tuple[str, str]]) -> List[Tuple[List[str], List[str]]]:
    return [(character_tokenize(noisy), character_tokenize(clean)) for noisy, clean in data]

if __name__ == "__main__":
    raw_data = load_data(goldset_dir)
    analyze_data(raw_data)
    processed_data = preprocess_data(raw_data)
    print(f"Processed {len(processed_data)} pairs")
    print("Sample processed pair:")
    print(f"Noisy: {processed_data[0][0][:50]}")
    print(f"Clean: {processed_data[0][1][:50]}")