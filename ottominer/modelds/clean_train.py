from pathlib import Path
import os
import sys
import re

rd = Path(__file__).resolve().parents[2]
data_dir = rd / 'corpus-texts'
txts_extracted_dir = data_dir / 'extracted_texts' # txt files extracted from pdfs -- will be used for training as noisy data
txts_clean_dir = data_dir / 'extracted_texts'
os.makedirs(txts_extracted_dir, exist_ok=True)
sys.path.append(str(rd))



input_dir = txts_extracted_dir
output_dir = txts_clean_dir


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
    
    sentences = re.split(r'(?<=[.?!:])\s+(?!\d+\.)', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return '\n'.join(sentences) 

def normalize_quotation(text):
    text = re.sub(r"(\b\w+)\s'(\w+)", r"\1'\2", text)
    return text

if __name__ == '__main__':
    text_files = [file for file in os.listdir(input_dir) if file.endswith('.txt')]
    
    for file_name in text_files:
        input_path = input_dir / file_name
        
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        cleaned_content = clean_text(content)
        
        cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content)
        
        output_path = output_dir / file_name
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        
        print(f"Processed and saved: {output_path}")