import re
from pathlib import Path
import os

# Define the input and output directories
rd = Path(__file__).resolve().parent
input_dir = rd / 'saved' / 'test_save'
output_dir = rd / 'saved' / 'final_saved'
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    # Remove numbers cojoined with words
    text = re.sub(r'([a-zA-ZğüşöçıİĞÜŞÖÇ]+)\d+', r'\1', text)
    
    # Reduce excessive punctuation cojoined with words to a single instance
    text = re.sub(r'([a-zA-ZğüşöçıİĞÜŞÖÇ]+)([!.,?;:]{2,})', lambda m: f"{m.group(1)}{m.group(2)[0]}", text)
    
    # Remove any excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = normalize_quotation(text)
    
    # Remove single characters/letters occurrences
    text = re.sub(r'\s+(\S)\s+', ' ', text)
    
    # Split sentences and place each on a new line, excluding numbered indices
    sentences = re.split(r'(?<=[.?!:])\s+(?!\d+\.)', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]  # Clean up extra whitespace
    return '\n'.join(sentences)  # Join sentences with new lines

def normalize_quotation(text):
    # Replace incorrect quotation marks with the correct format
    text = re.sub(r"(\b\w+)\s'(\w+)", r"\1'\2", text)
    return text

if __name__ == '__main__':
    # List all text files in the input directory
    text_files = [file for file in os.listdir(input_dir) if file.endswith('.txt')]
    
    # Process each text file
    for file_name in text_files:
        input_path = input_dir / file_name
        
        # Read the content of the file
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Clean the text using the clean_text function
        cleaned_content = clean_text(content)
        
        # Remove any extra gaps between lines
        cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content)
        
        # Define the output path and save the cleaned content
        output_path = output_dir / file_name
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        
        print(f"Processed and saved: {output_path}")