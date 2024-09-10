import os
import re
import logging
from pathlib import Path

logging.basicConfig(
    filename='text_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

bd_ = Path(__file__).resolve().parents[2]
input_file = bd_ / "corpus-texts" / "clean_corpus_text" / "merged_all.txt"
output_file = bd_ / "corpus-texts" / "merged_clean.txt"

def read_file_in_batches(file_path, batch_size=1024):
    """Read the file in batches to handle large files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                lines = file.readlines(batch_size)
                if not lines:
                    break
                yield lines
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise

def clean_text(text):
    """Clean text by removing unrecognized characters and reducing excessive whitespace."""
    try:
        allowed_chars = "abcdefghijklmnopqrstuvwxyzçğıöşüâîûABCDEFGHIJKLMNOPQRSTUVWXYZÇĞİÖŞÜÂÎÛ'.,;:?!()–"
        cleaned_text = re.sub(f'[^{allowed_chars} ]+', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        raise

def join_sentences(text):
    """Join lines that should form a single sentence."""
    try:
        text = re.sub(r'(?<![.!?])\n(?![.!?])', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.!?])\s+', r'\1\n', text)
        
        return text
    except Exception as e:
        logging.error(f"Error joining sentences: {e}")
        raise

def remove_short_lines(text):
    """Remove lines with fewer than two words or characters."""
    try:
        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line.split()) >= 2]
        return '\n'.join(filtered_lines)
    except Exception as e:
        logging.error(f"Error removing short lines: {e}")
        raise

def fix_structure(text):
    """Apply heuristic rules to fix the structure of the text."""
    try:
        text = join_sentences(text)
        text = remove_short_lines(text)
        return text
    except Exception as e:
        logging.error(f"Error fixing text structure: {e}")
        raise

def count_tokens(text):
    """Count tokens in the text."""
    tokens = text.split()
    return len(tokens)

def process_and_save_cleaned_text(input_path, output_path):
    """Process the text by cleaning and fixing its structure, then save to output file."""
    total_tokens_before = 0
    total_tokens_after = 0
    
    try:
        all_cleaned_text = []
        for batch in read_file_in_batches(input_path):
            for line in batch:
                total_tokens_before += count_tokens(line)
                cleaned = clean_text(line)
                structured = fix_structure(cleaned)
                total_tokens_after += count_tokens(structured)
                all_cleaned_text.append(structured)
                
        with open(output_path, 'w', encoding='utf-8') as output:
            output.write('\n'.join(all_cleaned_text))
        
        logging.info(f"Successfully cleaned and saved text to {output_path}")

        # Print token counts
        print(f"Total tokens before cleaning: {total_tokens_before}")
        print(f"Total tokens after cleaning: {total_tokens_after}")
        print(f"Tokens removed: {total_tokens_before - total_tokens_after}")
    
    except Exception as e:
        logging.error(f"Error processing text from {input_path} to {output_path}: {e}")

if __name__ == "__main__":
    process_and_save_cleaned_text(input_file, output_file)
    print(f"Processing completed. Cleaned text saved to {output_file}")
