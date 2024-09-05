import re
from pathlib import Path
import os
import fitz  # PyMuPDF
import glob
import numpy as np

rd_ = Path(__file__).resolve().parents[1]
input_dir = rd_ / 'corpus-texts' / 'pdfs'
od_ = rd_ / 'corpus-texts' / 'extracted_texts'
os.makedirs(od_, exist_ok=True)

class TextProcessor:
    def __init__(self, text: str):
        self.text = text

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

    def extract_sentences(self):
        """Extracts sentences that may span multiple lines."""
        pattern = r'([A-Z][^.?!]*?(?:\n\s*)+[^.?!]*[.?!])'
        sentences = re.findall(pattern, self.text, re.DOTALL)
        return [sentence.replace('\n', ' ').strip() for sentence in sentences]

    def process_text(self):
        """Processes the text by normalizing and extracting sentences."""
        self.normalize_text().join_hyphenated_words()
        return self.extract_sentences()

class PDFTextExtractor:
    def __init__(self, char_map):
        self.char_map = char_map

    def replace_characters(self, text):
        """Replaces specific characters based on the provided character map."""
        for old_char, new_char in self.char_map.items():
            text = text.replace(old_char, new_char)
        return text

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extracts text from PDF without layout parsing, handling multiple pages."""
        extracted_text = []
        document = fitz.open(pdf_path)

        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            extracted_text.append(page.get_text())
        
        document.close()
        return "\n".join(extracted_text)

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

    extractor = PDFTextExtractor(char_setup)
    pdf_files = glob.glob(str(input_dir) + '/*.pdf')
    
    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path}")
            content = extractor.extract_text_from_pdf(Path(pdf_path))
            
            if not content.strip():
                print(f"Warning: No content extracted from {pdf_path}. Skipping file.")
                continue
            
            normalized_content = extractor.replace_characters(content)
            text_processor = TextProcessor(normalized_content)
            formatted_text = "\n".join(text_processor.process_text())

            save_path = od_ / (Path(pdf_path).stem + '.txt')
            save_text_to_file(formatted_text, save_path)
            print(f"Successfully processed and saved: {save_path}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

if __name__ == "__main__":
    main()