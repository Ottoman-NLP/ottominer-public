import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from anim.progress import ProgressBar

class TextFilter:
    def __init__(self, input_dir, output_dir, progress_bar):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.progress_bar = progress_bar


    def normalize_turkish_characters(self, text):
        normalization_map = {
            """turkish character normalization map"""
            'â': 'a', 'Â': 'A',
            'î': 'i', 'Î': 'I',
            'û': 'u', 'Û': 'U',
            'ā': 'a', 'Ā': 'A'
        }
        return ''.join(normalization_map.get(char, char) for char in text)

    def remove_non_turkish_characters(self, text):
        """
        such regex patterns are used to filter the pdf text extracted from the pdf files.:
        - Arabic characters
        - Unwanted symbols
        - Bullet points
        - Dots in certain lines
        - Single characters
        - Citations
        - Hyphenated words
        - Titles
        """
        text = self.normalize_turkish_characters(text)
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

    def filter_text(self, input_file, output_file):

        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
            text = self.remove_non_turkish_characters(text)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Filtered text saved to {output_file}")

    def process_directory(self):

        """Process all text files in the input directory."""
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.txt')]
        total_files = len(files)

        if total_files == 0:
            print("No text files found in the directory.")
            return

        task = "Processing text files"
        t = self.progress_bar.start(task)

        for index, filename in enumerate(files):

            input_file = os.path.join(self.input_dir, filename)
            output_file = os.path.join(self.output_dir, filename)
            self.filter_text(input_file, output_file)
            progress = int(((index + 1) / total_files) * 100)
            self.progress_bar.update(task, progress)

        self.progress_bar.stop(t)

    @staticmethod
    def is_turkish(text):
        return bool(re.search(r'[a-zA-ZÇĞİıÖŞÜçğöşüâîûāÂÎÛĀ]', text))


if __name__ == "__main__":
    input_dir = "../texture/filtered_txt"
    output_dir = "../texture/final"
    os.makedirs(output_dir, exist_ok=True)
    progress_bar = ProgressBar() # custom and optional progress bar
    text_filter = TextFilter(input_dir, output_dir, progress_bar)
    text_filter.process_directory()