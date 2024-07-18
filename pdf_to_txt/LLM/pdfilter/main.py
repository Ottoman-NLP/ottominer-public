import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
import re
from anim.progress import ProgressBar # Custom progress bar

"""
this script uses PyMuPDF to extract text from PDF files.
It first analyzes the fonts used in the PDF file to determine the main font size and the next most used font size.
Then it extracts the text from the PDF file using the main font size and the next most used font size.
The extracted text is saved to a text file with the same name as the PDF file in the output directory.
There is normalization mapping for Arabic text to remove diacritics and other characters.

The most common font size is a proxy for the main font size used in the PDF file.
Please update file directory names as needed.
"""

def normalize_arabic(text):
    normalization_map = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
        'ى': 'ي', 'ئ': 'ي',
        'ة': 'ه',
        'ؤ': 'و',
        'ﺮﱡ': 'ﺮ',
        'ـ': '',  # Tatweel
        'ء': '',  # Hamza
        '،': '',

        # Diacritics
        'َ': '', 'ُ': '', 'ِ': '',
        'ً': '', 'ٌ': '', 'ٍ': '',
        'ّ': '', 'ﱢ': '',
        'ْ': '',

        # Corrected keys
        'رٰ': 'ر',
        'لٰ': 'ل', 'لٖ': 'ل',
        'غٰ': 'غ',
    }
    return ''.join(normalization_map.get(char, char) for char in text)

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def analyze_fonts(pdf_document):
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

def extract_text(pdf_path, main_font_size, next_size):
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

                        if is_arabic(text):
                            processing_arabic = True
                            text = text.strip("()")
                            normalized_text = normalize_arabic(text)
                            arabic_buffer.append(normalized_text)
                        elif processing_arabic:
                            if arabic_buffer:
                                arabic_text = " ".join(arabic_buffer)
                                extracted_text += "\n" + arabic_text[::-1] + "\n"
                                arabic_buffer = []
                            processing_arabic = False
                            line_text.append(text)
                        elif (font_size in [main_font_size, next_size] or (
                                is_bold and contains_alpha)) and not is_numeric_only_bold:
                            line_text.append(text)

                    if line_text:
                        line_content = " ".join(line_text)
                        extracted_text += line_content + "\n"

        if arabic_buffer:
            arabic_text = " ".join(arabic_buffer)
            extracted_text += "\n" + arabic_text[::-1] + "\n"
            arabic_buffer = []

    pdf.close()
    return extracted_text

def process_directory(input_dir, output_dir, progress_bar):
    files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    total_files = len(files)

    if total_files == 0:
        print("No PDF files found in the directory.")
        return

    task = "Processing PDF files"
    t = progress_bar.start(task)

    for index, filename in enumerate(files):
        file_path = os.path.join(input_dir, filename)
        pdf_document = fitz.open(file_path)
        main_font_size, next_size = analyze_fonts(pdf_document)
        if main_font_size:
            extracted_text = extract_text(file_path, main_font_size, next_size)
            output_path = os.path.join(output_dir, filename.replace('.pdf', '.txt'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"Processed {filename} and saved to {output_path}")
        progress = int(((index + 1) / total_files) * 100)
        progress_bar.update(task, progress)

    progress_bar.stop(t)


if __name__ == "__main__":
    # Hard-coded directories - change as needed
    input_dir = "../texture/filtered_txt"
    output_dir = "../texture/final"
    os.makedirs(output_dir, exist_ok=True)

    progress_bar = ProgressBar()
    process_directory(input_dir, output_dir, progress_bar)