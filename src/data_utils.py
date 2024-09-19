import torch
from collections import Counter
from typing import List, Tuple

class Vocabulary:
    def __init__(self, freq_threshold: int):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.word_freq = Counter()

    def build_vocabulary(self, sentences: List[List[str]]):
        for sentence in sentences:
            self.word_freq.update(sentence)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                self.add_word(word)

    def add_word(self, word: str):
        if word not in self.stoi:
            index = len(self.itos)
            self.itos[index] = word
            self.stoi[word] = index

    def numericalize(self, text: List[str]) -> List[int]:
        return [self.stoi.get(word, 0) for word in text]  # 0 is the index for <PAD>

def prepare_data(data: List[Tuple[List[str], List[str]]], freq_threshold: int) -> Tuple[Vocabulary, List[Tuple[List[int], List[int]]]]:
    vocab = Vocabulary(freq_threshold)
    noisy_sentences = [pair[0] for pair in data]
    clean_sentences = [pair[1] for pair in data]
    
    vocab.build_vocabulary(noisy_sentences + clean_sentences)
    
    processed_data = [
        (vocab.numericalize(noisy), vocab.numericalize(clean))
        for noisy, clean in data
    ]
    
    return vocab, processed_data

if __name__ == "__main__":
    from preprocess import load_data, goldset_dir

    data = load_data(goldset_dir)
    vocab, processed_data = prepare_data(data, freq_threshold=2)
    print(f"Vocabulary size: {len(vocab.itos)}")
    print(f"Processed data size: {len(processed_data)}")