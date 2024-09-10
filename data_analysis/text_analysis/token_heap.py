import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.data.path = [ # add nltk data to path -- I have to do this because I can't download the data otherwise
]

bd_ = Path(__file__).resolve().parents[3]
print(bd_)

input_file = bd_ / "corpus-texts" / "final_clean.txt"
output_file = bd_ / "corpus-texts" / "clean_text_analysis.txt"

try:
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file}")
    exit(1)
except IOError:
    print(f"Error: Unable to read input file at {input_file}")
    exit(1)

print("Attempting sentence tokenization...")
try:
    sentences = sent_tokenize(text)
    print(f"Successfully tokenized {len(sentences)} sentences.")
except LookupError as e:
    print(f"Error tokenizing sentences: {e}")
    print("Falling back to simple sentence splitting...")
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    print(f"Fallback tokenization: {len(sentences)} sentences.")

tokens = word_tokenize(text)

num_tokens = len(tokens)
num_sentences = len(sentences)

print(f"Number of tokens: {num_tokens}")
print(f"Number of sentences: {num_sentences}")

# Count the frequency of each token
token_counts = Counter(tokens)
num_unique_tokens = len(token_counts)
print(f"Number of unique tokens: {num_unique_tokens}")

k, beta = 10, 0.5
predicted_vocabulary_size = k * (num_tokens ** beta)
print(f"Predicted unique tokens (Heap's Law): {predicted_vocabulary_size:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_tokens + 1), [k * (n ** beta) for n in range(1, num_tokens + 1)], label="Heap's Law Prediction")
plt.axhline(y=num_unique_tokens, color='r', linestyle='-', label="Actual Unique Tokens")
plt.xlabel('Number of Tokens')
plt.ylabel('Vocabulary Size')
plt.title("Heap's Law Analysis")
plt.legend()

plot_file = bd_ / "corpus-texts" / "heaps_law_plot.png"
plt.savefig(plot_file)
print(f"Heap's Law plot saved to {plot_file}")

token_diversity = num_unique_tokens / num_tokens
print(f"Token diversity: {token_diversity:.4f}")

try:
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"Number of tokens: {num_tokens}\n")
        file.write(f"Number of sentences: {num_sentences}\n")
        file.write(f"Number of unique tokens: {num_unique_tokens}\n")
        file.write(f"Predicted unique tokens (Heap's Law): {predicted_vocabulary_size:.2f}\n")
        file.write(f"Token diversity: {token_diversity:.4f}\n")
    print(f"Analysis results written to {output_file}")
except IOError:
    print(f"Error: Unable to write to output file at {output_file}")

print("Sentiment Analysis: Not implemented - requires a custom model for Ottoman Turkish.")