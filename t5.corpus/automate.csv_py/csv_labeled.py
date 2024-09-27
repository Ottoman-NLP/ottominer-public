import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def compute_embeddings(model, sentences, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing embeddings"):
        batch = sentences[i:i+batch_size]
        batch_embeddings = model.module.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)


def align_sentences(noisy_embeddings, clean_embeddings, threshold=0.9, chunk_size=1000):
    device = noisy_embeddings.device
    m, n = noisy_embeddings.shape[0], clean_embeddings.shape[0]
    aligned_pairs = []

    for i in tqdm(range(0, m, chunk_size), desc="Aligning sentences"):
        chunk_noisy = noisy_embeddings[i:i+chunk_size]
        similarity_chunk = torch.matmul(chunk_noisy, clean_embeddings.T)
        
        for j in range(chunk_size):
            if i+j < m:
                best_match = torch.argmax(similarity_chunk[j]).item()
                reverse_best_match = torch.argmax(similarity_chunk[:, best_match]).item()
                if similarity_chunk[j, best_match] > threshold and reverse_best_match == j:
                    aligned_pairs.append((i+j, best_match))

    return aligned_pairs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model first
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    noisy_file = Path('corpus-texts/datasets/noisy.txt')
    clean_file = Path('corpus-texts/datasets/clean.txt')
    output_file = Path('corpus-texts/datasets/aligned_sentences.json')

    noisy_sentences = load_sentences(noisy_file)
    clean_sentences = load_sentences(clean_file)
    
    print("Computing embeddings for noisy sentences...")
    noisy_embeddings = compute_embeddings(model, noisy_sentences, device)
    print("Computing embeddings for clean sentences...")
    clean_embeddings = compute_embeddings(model, clean_sentences, device)


    print("Aligning sentences...")
    aligned_pairs = align_sentences(noisy_embeddings, clean_embeddings)

    aligned_data = [
        {"noisy": noisy_sentences[i], "clean": clean_sentences[j]}
        for i, j in aligned_pairs
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aligned_data, f, ensure_ascii=False, indent=2)

    print(f"Aligned sentences saved to {output_file}")
    print(f"Total aligned pairs: {len(aligned_data)}")

if __name__ == "__main__":
    main()