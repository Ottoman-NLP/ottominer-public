import pymupdf4llm
import os
import re
import unicodedata
from pathlib import Path

class OttomanPDFMiner:
    def __init__(self, input_pdf):
        self.input_pdf = input_pdf
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
    def extract_and_process(self):
        # Extract text in markdown format with page chunks
        md_text = pymupdf4llm.to_markdown(
            self.input_pdf,
            page_chunks=True,
            write_images=True
        )
        
        # Process each page chunk
        processed_chunks = []
        for chunk in md_text:
            processed_text = self._process_ottoman_text(chunk['text'])
            processed_chunks.append(processed_text)
            
        return '\n\n'.join(processed_chunks)
    
    def _process_ottoman_text(self, text):
        # Ottoman-specific character normalization
        ottoman_mappings = {
            'À': 'â', 'Á': 'â', 'è': 'i', 'ì': 'î',
            'ò': 'ô', 'ù': 'û', 'é': 'e', 'õ': 'ö',
            'ÿ': 'ü', 'È': 'i', 'Ì': 'î', 'æ': 'a',
            'ó': 'ô', 'ú': 'û', 'í': 'î', 'à': 'â',
            'èİ': 'i', 'ée': 'e', 'èÿ': 'ü'
        }
        
        # Apply Ottoman-specific normalizations
        for old, new in ottoman_mappings.items():
            text = text.replace(old, new)
            
        # Handle special Ottoman text patterns
        text = re.sub(r'(?<=\w)-zÀde', 'zade', text)
        text = re.sub(r'(?<=\w)é(?=\w)', 'e', text)
        
        # Clean and format
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[(\d+)\]', r'\n[\1]\n', text)
        
        return text.strip()
    
    def save_output(self, processed_text):
        output_path = self.script_dir / "output_cleaned.md"
        output_path.write_text(processed_text, encoding='utf-8')
        return output_path

def main():
    pdf_path = "input.pdf"
    miner = OttomanPDFMiner(pdf_path)
    processed_text = miner.extract_and_process()
    output_path = miner.save_output(processed_text)
    print(f"Processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
