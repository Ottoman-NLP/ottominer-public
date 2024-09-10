import re
import logging
from pathlib import Path

logging.basicConfig(
    filename='post_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

bd_ = Path(__file__).resolve().parents[2]
input_file = bd_ / "corpus-texts" / "merged_clean.txt"
output_file = bd_ / "corpus-texts" / "final_clean.txt"

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise

def reformat_sentences(text):
    try:
        text = re.sub(r'(?<![.!?])\n(?![.!?])', ' ', text)
        
        text = re.sub(r'([.!?])\s+', r'\1\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n\s+', '\n', text)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    except Exception as e:
        logging.error(f"Error reformating sentences: {e}")
        raise

def re_segment_sentences(text):
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return '\n'.join(sentences)
    except Exception as e:
        logging.error(f"Error re-segmenting sentences: {e}")
        raise

def count_lines_and_tokens(text):
    lines = text.split('\n')
    num_lines = len(lines)
    num_tokens = sum(len(line.split()) for line in lines)
    return num_lines, num_tokens

def save_cleaned_text(text, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as output:
            output.write(text)
        logging.info(f"Successfully saved formatted text to {output_path}")
    except Exception as e:
        logging.error(f"Error saving file {output_path}: {e}")
        raise

def post_process_text(input_path, output_path):
    try:
        original_text = read_file(input_path)
        
        initial_lines, initial_tokens = count_lines_and_tokens(original_text)

        formatted_text = reformat_sentences(original_text)

        formatted_text = re_segment_sentences(formatted_text)

        final_lines, final_tokens = count_lines_and_tokens(formatted_text)

        save_cleaned_text(formatted_text, output_path)
        
        print(f"Initial number of lines: {initial_lines}")
        print(f"Initial number of tokens: {initial_tokens}")
        print(f"Final number of lines: {final_lines}")
        print(f"Final number of tokens: {final_tokens}")
        print(f"Lines removed: {initial_lines - final_lines}")
        print(f"Tokens removed: {initial_tokens - final_tokens}")
        
        logging.info(f"Post-processing completed. Final cleaned text saved to {output_path}")
    except Exception as e:
        logging.error(f"Error in post-processing text from {input_path} to {output_path}: {e}")

if __name__ == "__main__":
    post_process_text(input_file, output_file)
    print(f"Post-processing completed. Final cleaned text saved to {output_file}")
