import os
import re
from pathlib import Path
import fitz
import sys
rd = Path(__file__).resolve().parents[2]
sys.path.append(str(rd))

input_dir = rd / "LLM" / "texture" / "data_pdfs"
output_dir = rd / "LLM" / "texture" / "texts" 
from ..ottominer.anim.progress import ProgressBar

print(f"Root directory: {rd}")
os.makedirs(output_dir, exist_ok=True)
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")


class TextFilter:
    def __init__(self, input_dir, output_dir, progress_bar):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.progress_bar = progress_bar

    @staticmethod
    def normalize_chars(text):
        normalization_map = {
            'â': 'a', 'î': 'i', 'û': 'u',
            'Â': 'A', 'Î': 'I', 'Û': 'U',
            'ā': 'a', 'Ā': 'A'
        }

        return ''.join(normalization_map.get(char, char) for char in text)

    @staticmethod
    def split_into_sentences(text):
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        sentences = sentence_endings.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def clean_text(text):
        arabic_or_diacritics = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        pua_characters = re.compile(r'[\uE000-\uF8FF]')
        cleaned_lines = []

        lines = text.splitlines()
        for i, line in enumerate(lines):
            if arabic_or_diacritics.search(line) or pua_characters.search(line):
                continue
            if len(line) == 1 and not line.isalnum():
                continue
            if re.match(r'^\d+$', line):
                continue

            # Handle hyphenated words at line breaks
            if line.endswith('-') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[a-zA-Z]', next_line):
                    line = line[:-1] + next_line
                    lines[i + 1] = ''  # Remove the next line since it's joined

            cleaned_lines.append(line)

        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = TextFilter.normalize_chars(cleaned_text)
        
        return cleaned_text

    @staticmethod
    def clean_text_file(file_path, output_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        cleaned_text = TextFilter.clean_text(text)
        sentences = TextFilter.split_into_sentences(cleaned_text)
        
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for sentence in sentences:
                if re.search(r'\w', sentence):
                    output_file.write(sentence + '\n')

    def filter_text(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
            text = self.remove_non_turkish_characters(text)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Filtered text saved to {output_file}")

    def remove_non_turkish_characters(self, text):
        normalization_map = {
            "“": '"', "”": '"', "’": "'", "–": "-", "â": "a",'ā': 'a', "ê": "e", "î": "i",
            "ô": "ö", "û": "ü", "ī": "i", "ū": "ü", "ō": "o", "ē": "e", "ā": "a",
            "Â": "A", "Ê": "E", "Î": "İ", "Ô": "Ö", "Û": "Ü", "Ī": "İ", "Ū": "Ü",
            "Ō": "Ö", "Ē": "E",'Â': 'A', "Ā": "A", "": "", "": "", "œ": "i",
            "†": "u", "å": "a", "¢": "i", "": "", "": "",
            'î': 'i', 'û': 'u', 'Î': 'I', 'Û': 'U',
        }
        text = ''.join(normalization_map.get(char, char) for char in text)
        lines = text.split('\n')
        cleaned_lines = []

        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        unwanted_symbols_pattern = re.compile(r'[()،؛,]')
        bullet_point_pattern = re.compile(r'^\s*[-iİıI]')
        dots_pattern = re.compile(r'^\s*\.*\s*$')
        single_char_pattern = re.compile(r'^\s*[a-zA-Z0-9ş]\s*$')
        citation_pattern = re.compile(r'^\s*\[\s*\d*\s*\]\s*$')
        hyphenated_word_pattern = re.compile(r'-$')
        title_pattern = re.compile(r'^[A-Z\s\'.-]+$')

        i = 0
        while i < len(lines):
            line = lines[i]
            cleaned_line = arabic_pattern.sub('', line).strip()
            cleaned_line = unwanted_symbols_pattern.sub('', cleaned_line).strip()

            if bullet_point_pattern.match(cleaned_line):
                i += 1
                continue
            if dots_pattern.match(cleaned_line):
                i += 1
                continue
            if single_char_pattern.match(cleaned_line):
                i += 1
                continue
            if citation_pattern.match(cleaned_line):
                i += 1
                continue

            if hyphenated_word_pattern.search(cleaned_line) and i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                next_word_match = re.match(r'^\s*(\S+)', next_line)
                if next_word_match and self.is_turkish(next_word_match.group(1)):
                    cleaned_line = cleaned_line.rstrip('-') + next_word_match.group(1)
                    lines[i + 1] = next_line[next_word_match.end():].lstrip()

            if title_pattern.match(cleaned_line):
                cleaned_line = f"\n----\n{cleaned_line}\n----\n"

            if cleaned_line:
                cleaned_lines.append(cleaned_line)

            i += 1

        filtered_lines = []
        for i, line in enumerate(cleaned_lines):
            filtered_lines.append(line)
            if line == "" and i < len(cleaned_lines) - 1 and cleaned_lines[i + 1] != "":
                filtered_lines.append("")

        return '\n'.join(filtered_lines)

    def analyze_fonts(self, pdf_document):
        font_data = {}
        for page in pdf_document:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block.keys():
                    lines = block['lines']
                    for line in lines:
                        spans = line['spans']
                        for span in spans:
                            font_size = span['size']
                            if font_size in font_data:
                                font_data[font_size] += 1
                            else:
                                font_data[font_size] = 1

        if font_data:
            sorted_fonts = sorted(font_data.items(), key=lambda item: item[1], reverse=True)
            main_font_size = sorted_fonts[0][0]
            next_size = sorted_fonts[1][0] if len(sorted_fonts) > 1 else main_font_size
            return main_font_size, next_size
        return None, None

    def extract_text(self, pdf_path, main_font_size, next_size):
        pdf = fitz.open(pdf_path)
        extracted_text = ""
        arabic_buffer = []
        processing_arabic = False

        for page in pdf:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block.keys():
                    lines = block['lines']
                    for line in lines:
                        line_text = []
                        spans = line['spans']
                        for span in spans:
                            text = span['text'].strip()
                            font_size = span['size']
                            is_bold = 'Bold' in span['font'] or 'Black' in span['font']
                            contains_alpha = any(c.isalpha() for c in text)
                            is_numeric_only_bold = is_bold and text.isnumeric()

                            if self.is_turkish(text):
                                processing_arabic = True
                                text = text.strip("()")
                                normalized_text = self.normalize_chars(text)
                                arabic_buffer.append(normalized_text)
                            elif processing_arabic:
                                if arabic_buffer:
                                    arabic_text = " ".join(arabic_buffer)
                                    extracted_text += "\n" + arabic_text[::-1] + "\n"
                                    arabic_buffer = []
                                processing_arabic = False
                                line_text.append(text)
                            elif (font_size in [main_font_size, next_size] and contains_alpha and not is_numeric_only_bold):
                                line_text.append(text)

                        extracted_text += " ".join(line_text) + "\n"

        if arabic_buffer:
            extracted_text += "\n" + " ".join(arabic_buffer[::-1]) + "\n"

        pdf.close()
        return extracted_text

    def process_pdf_directory(self):
        if not self.input_dir.exists():
            print(f"Input directory does not exist: {self.input_dir}")
            return

        files = list(self.input_dir.rglob('*.pdf'))
        total_files = len(files)

        if total_files == 0:
            print("No PDF files found in the directory or its subdirectories.")
            return

        task = "Processing PDF files"
        next_target = "Next file "
        t = self.progress_bar.start(task, next_target)

        for index, file_path in enumerate(files):
            main_font_size, next_size = self.analyze_fonts(file_path)
            if main_font_size:
                extracted_text = self.extract_text(file_path, main_font_size, next_size)
                output_path = self.output_dir / (file_path.stem + '.txt')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"Processed {file_path.name} and saved to {output_path}")
            progress = int(((index + 1) / total_files) * 100)
            self.progress_bar.update(task, progress, next_target)

        self.progress_bar.stop(t)

    def clean_directory(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.output_dir):
            if filename.endswith(".txt"):
                input_path = self.output_dir / filename
                output_path = self.output_dir / filename
                self.clean_text_file(input_path, output_path)
                print(f"Processed {filename}")

def main():
    root_dir = Path(__file__).resolve().parents[2]
    input_dir = root_dir / "texture" / "texts"
    output_dir = root_dir / "texture" / "txts"
    os.makedirs(output_dir, exist_ok=True)
    progress_bar = ProgressBar()
    text_filter = TextFilter(input_dir, output_dir, progress_bar)
    text_filter.process_pdf_directory()
    text_filter.clean_directory()

if __name__ == "__main__":
    main()
