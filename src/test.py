import torch
import os
from model import create_model
from data_utils import prepare_data, Vocabulary
from preprocess import load_data, goldset_dir
from nltk.translate.bleu_score import corpus_bleu
from collections import Counter
import matplotlib.pyplot as plt

def correct_sentence(model, sentence, vocab, device):
    model.eval()
    tokens = [vocab.stoi.get(token, vocab.stoi['<unk>']) for token in sentence.split()]
    
    if len(tokens) == 0:
        print(f"Warning: Empty sentence: '{sentence}'")
        return sentence  # Return the original sentence if it's empty
    
    source = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            output = model(source, source)
            predicted = output.argmax(2).squeeze(0)
            corrected = [vocab.itos[token.item()] for token in predicted]
            return ' '.join(corrected)
        except RuntimeError as e:
            print(f"Error processing sentence: {e}")
            return sentence  # Return the original sentence if there's an error


def correct_file(model, input_file, output_file, vocab, device):
    corrected_lines = []
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                corrected_line = correct_sentence(model, line.strip(), vocab, device)
                f_out.write(corrected_line + '\n')
                corrected_lines.append(corrected_line)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                print(f"Problematic line: '{line.strip()}'")
                f_out.write(line)  # Write the original line if there's an error
                corrected_lines.append(line.strip())
    return corrected_lines

def calculate_bleu(reference_file, candidate_lines):
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip().split() for line in f]
    candidates = [line.split() for line in candidate_lines]
    return corpus_bleu([[ref] for ref in references], candidates)

def exploratory_data_analysis(original_file, corrected_lines):
    # Read original file
    with open(original_file, 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f]

    # Calculate statistics
    original_lengths = [len(line.split()) for line in original_lines]
    corrected_lengths = [len(line.split()) for line in corrected_lines]

    original_vocab = Counter(word for line in original_lines for word in line.split())
    corrected_vocab = Counter(word for line in corrected_lines for word in line.split())

    # Plot length distributions
    plt.figure(figsize=(10, 5))
    plt.hist(original_lengths, bins=20, alpha=0.5, label='Original')
    plt.hist(corrected_lengths, bins=20, alpha=0.5, label='Corrected')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths')
    plt.legend()
    plt.savefig('sentence_length_distribution.png')
    plt.close()

    # Plot vocabulary size
    plt.figure(figsize=(10, 5))
    plt.bar(['Original', 'Corrected'], [len(original_vocab), len(corrected_vocab)])
    plt.ylabel('Vocabulary Size')
    plt.title('Vocabulary Size Comparison')
    plt.savefig('vocabulary_size_comparison.png')
    plt.close()

    return {
        'avg_original_length': sum(original_lengths) / len(original_lengths),
        'avg_corrected_length': sum(corrected_lengths) / len(corrected_lengths),
        'original_vocab_size': len(original_vocab),
        'corrected_vocab_size': len(corrected_vocab)
    }
def main():
    # Load data and prepare vocabulary
    raw_data = load_data(goldset_dir)
    vocab, _ = prepare_data(raw_data, freq_threshold=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    input_size = len(vocab.itos)
    output_size = len(vocab.itos)
    model = create_model(input_size, output_size, device)

    # Load trained model weights
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # Process file
    input_file = os.path.join('corpus-texts', 'MT_testset', 'original.txt')
    output_file = os.path.join('corpus-texts', 'MT_testset', 'corrected.txt')
    corrected_lines = correct_file(model, input_file, output_file, vocab, device)

    print(f"Corrected text has been written to {output_file}")

    # Comment out BLEU score calculation if reference file is not available
    # reference_file = os.path.join('corpus-texts', 'MT_testset', 'reference.txt')
    # bleu_score = calculate_bleu(reference_file, corrected_lines)
    # print(f"BLEU Score: {bleu_score}")

    # Perform exploratory data analysis
    eda_results = exploratory_data_analysis(input_file, corrected_lines)
    print("Exploratory Data Analysis Results:")
    print(f"Average Original Sentence Length: {eda_results['avg_original_length']:.2f}")
    print(f"Average Corrected Sentence Length: {eda_results['avg_corrected_length']:.2f}")
    print(f"Original Vocabulary Size: {eda_results['original_vocab_size']}")
    print(f"Corrected Vocabulary Size: {eda_results['corrected_vocab_size']}")
    print("Plots have been saved as 'sentence_length_distribution.png' and 'vocabulary_size_comparison.png'")

if __name__ == "__main__":
    main()