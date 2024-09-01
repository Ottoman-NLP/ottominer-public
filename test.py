import re
from pathlib import Path
import os
import fitz  # PyMuPDF
import glob
import layoutparser as lp
from layoutparser.models import PaddleDetectionLayoutModel
import numpy as np

# Define the input and output directories
rd = Path(__file__).resolve().parent
input_dir = rd / 'saved' / 'pdfs'
save_dir = rd / 'saved' / 'test_save'
os.makedirs(save_dir, exist_ok=True)

class RegExFinder:
    def __init__(self, text: str):
        self.text = text

    def regex_pattern(self) -> str:
        # Pattern to capture sentences that span multiple lines, considering common sentence terminators.
        line_pattern = r'([A-Z][^.?!]*?(?:\n\s*)+[^.?!]*[.?!])'
        return line_pattern

    def remove_unwanted_characters(self) -> str:
        # Normalize excessive punctuation, keeping only one instance and ensuring it's attached to the word
        self.text = re.sub(r'([a-zA-ZğüşöçıİĞÜŞÖÇ]+)([!.,?;:]{2,})', lambda m: f"{m.group(1)}{m.group(2)[0]}", self.text)
        self.text = re.sub(r'[\[\];]', '', self.text)  # Remove unwanted characters like brackets and semicolons
        self.text = re.sub(r'\b\w[.?!]\s*', '', self.text, flags=re.MULTILINE)  # Remove lines ending with a single character followed by punctuation
        return self.text

    def join_hyphenated_words(self) -> str:
        # Join words that are split by a hyphen at the end of the line with the next line
        self.text = re.sub(r'-\s+', '', self.text)  # Remove hyphen followed by space
        return self.text

    def collect_sentences(self) -> list:
        self.text = self.remove_unwanted_characters()
        self.text = self.join_hyphenated_words()
        pattern = self.regex_pattern()
        sentences = re.findall(pattern, self.text, re.DOTALL)
        # Replacing multiple newline characters with a space to form complete sentences on a single line.
        return [sentence.replace('\n', ' ').strip() for sentence in sentences]

class TestText:
    def _format_text(self, text: RegExFinder):
        sentences = text.collect_sentences()
        formatted_text = '\n'.join(sentences)
        return formatted_text

char_setup = {
    'â': 'a', 'Â': 'A',
    'ā': 'a', 'Ā': 'A',
    "ê": "e",
    "ē": "e", "Ē": "E",
    "å": "a", "Å": "A",
    'î': 'i', 'Î': 'I',
    "ī": "i", "Ī": "I", 
    'û': 'u', 'Û': 'U',
    "ô": "ö",
    "ō": "o",  "Ō": "Ö",
}

# Load a pre-trained model for layout analysis (using PaddleDetection in this example)
model = PaddleDetectionLayoutModel(
    'lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config',
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config={"USE_GPU": True, "LIMIT_BATCH_SIZE_1": True}  # Corrected to use a dictionary
)

def extract_text_with_layoutparser(pdf_path: Path) -> str:
    print(f"Processing: {pdf_path}")
    extracted_text = ""
    document = fitz.open(pdf_path)
    
    for page_num in range(document.page_count):
        print(f"Analyzing page {page_num + 1} of {document.page_count}")
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Perform layout analysis
        layout = model.detect(image)
        
        # Filter out elements that are not the main text (ignoring headers, footers, and side notes)
        text_blocks = lp.Layout([b for b in layout if b.type == 'Text' and 
                                 0.15 < b.block.y_1 / page.rect.height < 0.85])  # Keep blocks in the main body region

        # Extract and concatenate text from text blocks
        for block in text_blocks:
            # Convert the block coordinates to a PyMuPDF Rect object
            rect = fitz.Rect(block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2)
            extracted_text += page.get_text(clip=rect) + "\n"
    
    document.close()
    print(f"Finished processing {pdf_path}")
    return extracted_text

def main():
    pdf_files = glob.glob(str(input_dir) + '/*.pdf')
    
    for pdf_path in pdf_files:
        try:
            content = extract_text_with_layoutparser(Path(pdf_path))
            
            if not content.strip():
                print(f"Warning: No content extracted from {pdf_path}.")
                continue
            
            # Normalize characters in the content
            for old_char, new_char in char_setup.items():
                content = content.replace(old_char, new_char)
            
            regex_finder = RegExFinder(content)
            formatted_text = TestText()._format_text(regex_finder)

            # Define the output path and save the formatted text
            save_path = save_dir / (Path(pdf_path).stem + '.txt')
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(formatted_text)
            
            print(f"Processed and saved: {save_path}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

if __name__ == "__main__":
    main()
    
# main docs are saved multiple folders to increase consistent analyse 
