import re
from pathlib import Path
import os

# Define the input and output directories
root_dir = Path(__file__).resolve().parents[1]
input_dir = root_dir / 'corpus-texts' / 'extracted_texts'
output_dir = root_dir / 'corpus-texts' / 'clean_texts'
os.makedirs(output_dir, exist_ok=True)

class TextCleaner:
    def __init__(self, text: str, char_map: dict):
        self.text = text
        self.char_map = char_map

    def normalize_characters(self):
        """Replaces specific characters based on the provided character map."""
        for old_char, new_char in self.char_map.items():
            self.text = self.text.replace(old_char, new_char)
        return self

    def normalize_text(self):
        """Removes unwanted characters and normalizes punctuation."""
        self.text = re.sub(r'([a-zA-ZğüşöçıİĞÜŞÖÇ]+)([!.,?;:]{2,})', lambda m: f"{m.group(1)}{m.group(2)[0]}", self.text)
        self.text = re.sub(r'[\[\];]', '', self.text)
        self.text = re.sub(r'\b\w[.?!]\s*', '', self.text, flags=re.MULTILINE)
        return self

    def join_hyphenated_words(self):
        """Joins words split by hyphens at line endings."""
        self.text = re.sub(r'-\s+', '', self.text)
        return self

    def clean_text(self):
        """Cleans the text by normalizing characters, punctuation, and formatting."""
        # Remove numbers joined with words
        self.text = re.sub(r'([a-zA-ZğüşöçıİĞÜŞÖÇ]+)\d+', r'\1', self.text)
        
        # Normalize quotation marks and punctuation
        self.text = re.sub(r"(\b\w+)\s'(\w+)", r"\1'\2", self.text)
        
        # Reduce excessive spaces
        self.text = re.sub(r'\s+', ' ', self.text).strip()
        
        # Remove single character occurrences surrounded by spaces
        self.text = re.sub(r'\s+(\S)\s+', ' ', self.text)
        
        # Split sentences and keep each on a new line
        sentences = re.split(r'(?<=[.?!:])\s+(?!\d+\.)', self.text)
        sentences = [sentence.strip() for sentence in sentences if sentence]
        return '\n'.join(sentences)

def save_text_to_file(text, path):
    """Saves processed text to a specified path."""
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)

def main():
    char_setup = {
        'â': 'a', 'Â': 'A', 'ā': 'a', 'Ā': 'A', "ê": "e", "ē": "e", "Ē": "E",
        "å": "a", "Å": "A", 'î': 'i', 'Î': 'I', "ī": "i", "Ī": "I", 'û': 'u', 
        'Û': 'U', "ô": "ö", "ō": "o", "Ō": "Ö",
    }

    # List all text files in the input directory
    text_files = [file for file in os.listdir(input_dir) if file.endswith('.txt')]
    
    # Process each text file
    for file_name in text_files:
        input_path = input_dir / file_name
        
        # Read the content of the file
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Create a TextCleaner instance, normalize characters, and clean the text
        cleaner = TextCleaner(content, char_setup)
        cleaned_content = cleaner.normalize_characters().normalize_text().join_hyphenated_words().clean_text()
        
        # Preserve sentence order with one sentence per line
        cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content)
        
        # Define the output path and save the cleaned content
        output_path = output_dir / f"{Path(file_name).stem}_clean.txt"
        save_text_to_file(cleaned_content, output_path)
        
        print(f"Processed and saved: {output_path}")

if __name__ == '__main__':
    main()