from pathlib import Path
import os
import fitz  # PyMuPDF
import glob
import re

# Define input and output directories
root_dir = Path(__file__).resolve().parents[1]
input_dir = root_dir / 'corpus-texts' / 'pdfs'
output_dir = root_dir / 'corpus-texts' / 'extracted_texts'
os.makedirs(output_dir, exist_ok=True)

class PDFTextExtractor:
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
    pdf_files = glob.glob(str(input_dir) + '/*.pdf')
    extractor = PDFTextExtractor()

    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path}")
            content = extractor.extract_text_from_pdf(Path(pdf_path))
            
            if not content.strip():
                print(f"Warning: No content extracted from {pdf_path}. Skipping file.")
                continue

            # Normalize and ensure each sentence is on a unique line
            content = content.replace('\n', ' ').strip()  # Flatten the content first
            sentences = [sentence.strip() for sentence in content.split('.') if sentence]  # Split by periods, trim spaces
            formatted_text = '.\n'.join(sentences) + '.'  # Rejoin sentences with each ending in a period and on a new line
            
            # Define the output path with the new naming convention
            save_path = output_dir / f"{Path(pdf_path).stem}_extract.txt"
            save_text_to_file(formatted_text, save_path)
            print(f"Successfully processed and saved: {save_path}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

if __name__ == "__main__":
    main()