import sentencepiece as spm
import re
import json
from pathlib import Path
import logging
import unicodedata
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = Path(__file__).resolve().parents[2]
input_path = base_dir / 'corpus-texts' / 'train_data' / 'labeled_data.json'
output_dir = base_dir / 'corpus-texts' / 'train_data'
test_path = base_dir / 'corpus-texts' / 'test_data' / 'test_data.json'
output_dir.mkdir(parents=True, exist_ok=True)

def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s\']', '', text)
    text = re.sub(r'(ler|lar|dir|dır|dür|dur)(?=\s|$)', r' \1', text)
    text = re.sub(r'\b(bi|na|la)', r'\1 ', text)
    text = re.sub(r'\b\w(\s\w)+\b', lambda m: m.group(0).replace(' ', ''), text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_test_data(sp, test_path):
    total_words = 0
    total_tokens = 0
    token_lengths = []
    word_to_token_ratios = []

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        for item in test_data:
            clean_text = preprocess_text(item['clean'])
            words = clean_text.split()
            total_words += len(words)
            tokens = sp.encode(clean_text, out_type=str)
            total_tokens += len(tokens)
            token_lengths.extend([len(token) for token in tokens])
            word_to_token_ratios.append(len(tokens) / len(words) if words else 0)

    return total_words, total_tokens, token_lengths, word_to_token_ratios

def calculate_score(avg_tokens_per_word, avg_token_length, avg_word_to_token_ratio):
    ideal_tokens_per_word = 1.5
    ideal_token_length = 4.0
    ideal_word_to_token_ratio = 1.3
    
    weight_tokens_per_word = 0.4
    weight_token_length = 0.3
    weight_word_to_token_ratio = 0.3
    
    score_tokens_per_word = max(0, 1 - abs(avg_tokens_per_word - ideal_tokens_per_word) / ideal_tokens_per_word)
    score_token_length = max(0, 1 - abs(avg_token_length - ideal_token_length) / ideal_token_length)
    score_word_to_token_ratio = max(0, 1 - abs(avg_word_to_token_ratio - ideal_word_to_token_ratio) / ideal_word_to_token_ratio)
    
    overall_score = (score_tokens_per_word * weight_tokens_per_word +
                     score_token_length * weight_token_length +
                     score_word_to_token_ratio * weight_word_to_token_ratio)
    
    return overall_score * 100

try:

    logging.info("Loading and preprocessing data...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"Using {len(data)} items from the dataset")
    clean_texts = [preprocess_text(item['clean']) for item in data]
    logging.info("Data loaded and preprocessed successfully.")

    logging.info("Preparing to train SentencePiece model...")
    with open('temp_input.txt', 'w', encoding='utf-8') as f:
        for text in clean_texts:
            f.write(text + '\n')
    
    logging.info("Starting SentencePiece training...")
    spm.SentencePieceTrainer.train(input='temp_input.txt',
                                model_prefix='ottoman_sp',
                                vocab_size=40000,
                                character_coverage=1.0,
                                model_type='bpe',
                                input_sentence_size=1000000,
                                shuffle_input_sentence=True,
                                split_by_whitespace=True,
                                split_by_unicode_script=False,
                                treat_whitespace_as_suffix=True,
                                max_sentencepiece_length=64,
                                pad_id=3,
                                unk_id=0,
                                bos_id=1,
                                eos_id=2,
                                user_defined_symbols=['<PAD>', '<UNK>', '<BOS>', '<EOS>']
                                )
    logging.info("SentencePiece training completed.")


    sp = spm.SentencePieceProcessor()
    sp.load('ottoman_sp.model')
    logging.info("SentencePiece model loaded successfully.")
    logging.info("Processing test data...")
    total_words, total_tokens, token_lengths, word_to_token_ratios = process_test_data(sp, test_path)
    logging.info("Test data processed successfully.")

    logging.info("Calculating statistics...")
    avg_tokens_per_word = total_tokens / total_words
    avg_token_length = sum(token_lengths) / len(token_lengths)
    avg_word_to_token_ratio = sum(word_to_token_ratios) / len(word_to_token_ratios)

    print(f"\nStatistics:")
    print(f"Total words: {total_words}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per word: {avg_tokens_per_word:.2f}")
    print(f"Average token length: {avg_token_length:.2f}")
    print(f"Average word-to-token ratio: {avg_word_to_token_ratio:.2f}")

    logging.info("Calculating overall score...")
    overall_score = calculate_score(avg_tokens_per_word, avg_token_length, avg_word_to_token_ratio)
    print(f"\nOverall Tokenization Score: {overall_score:.2f}%")

    if overall_score >= 80:
        print("Tokenization quality: GOOD")
        print("Recommendation: Continue with this tokenization approach and focus on improving the model.")
    else:
        print("Tokenization quality: NEEDS IMPROVEMENT")
        print("Recommendation: Further refine the tokenization process or consider more advanced techniques.")

    logging.info("Script completed successfully.")

except Exception as e:
    logging.error(f"Error during script execution: {str(e)}")
    raise