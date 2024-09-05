from pathlib import Path
import os
import sys
import re
from collections import Counter

rd = Path(__file__).resolve().parents[2]
data_dir = rd / 'corpus-texts' 
pdfs_dir = data_dir / 'pdfs'
txts_extracted_dir = data_dir / 'extracted_texts'
txts_clean_dir = data_dir / 'clean_texts'
os.makedirs(txts_extracted_dir, exist_ok=True)
os.makedirs(txts_clean_dir, exist_ok=True)
sys.path.append(str(rd))

pbdir = rd / "ottominer-public" / "ottominer" / "anim"
sys.path.append(str(pbdir))


input_dir = txts_extracted_dir
output_dir = txts_clean_dir

def normalize_chars(text):
    normalization_map = {
        'â': 'a',
        'î': 'i',
        'ô': 'o',
        'û': 'u',
        'Â': 'A',
        'Î': 'I',
        'ī': 'ı',
        'Ô': 'O',
        'Û': 'U',
        'ū': 'u',
        'Ū': 'U',
        'Ī': 'I',
        'ā': 'a',
        'Ā': 'A'
    }
    for char in normalization_map:
        text = text.replace(char, normalization_map[char])
    return text

def split_into_sentences(text):
    # Split sentences on ., ?, !, :, and ." followed by a space
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|:)\s')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def clean_text(text):
    arabic_or_diacritics = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    pua_characters = re.compile(r'[\uE000-\uF8FF]')
    cleaned_lines = []

    lines = text.splitlines()
    for i, line in enumerate(lines):

        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'\s*[*"\']\s*', '', line)
        line = re.sub(r'\s*[=§]\s*', '', line)
        line = re.sub(r'\s*----+\s*', '', line)

        # Handle "Aded: NUMBER" case
        if re.search(r'\bAded:\s*$', line):
            if i + 1 < len(lines) and re.match(r'^\d+$', lines[i + 1].strip()):
                line = line.strip() + lines[i + 1].strip()
                lines[i + 1] = ''
                
        line = re.sub(r'(\bSEBILÜRREŞAD\s*)?\b(CİLD|ADED|SAYFA)\b\s*\d+(-\d+)?', '', line)

        if arabic_or_diacritics.search(line) or pua_characters.search(line):
            continue
        if len(line) == 1 and not line.isalnum():
            continue
        if re.match(r'^\d+$', line):
            continue
        if re.match(r'^[A-ZİĞÜŞÖÇ]+\s*$', line) or re.match(r'^[A-ZİĞÜŞÖÇ\-]+$', line) or re.match(r'^[A-ZİĞÜŞÖÇ]+(?:-[a-z]+)?$', line):
            # This line is a potential title
            title_lines = [line]
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line.isupper() or re.match(r'^[A-ZİĞÜŞÖÇ]+(?:-[a-z]+)?$', next_line):
                    title_lines.append(next_line)
                    lines[j] = ''  # Remove the next line since it's joined
                else:
                    break
            title = ' '.join(title_lines)
            cleaned_lines.append(f"Başlık: {title}")
            continue


        if line.endswith('-') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^[a-zA-Z]', next_line):
                line = line[:-1] + next_line
                lines[i + 1] = ''  # Remove the next line since it's joined

        # Join words with spaces before '''

        cleaned_lines.append(line)

    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = normalize_chars(cleaned_text)
    
    return cleaned_text

def clean_text_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    cleaned_text = clean_text(text)
    sentences = split_into_sentences(cleaned_text)
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for sentence in sentences:
            if re.search(r'\w', sentence):  # Ensure the sentence contains at least one word character
                output_file.write(sentence + '\n')

def clean_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_tokens = 0
    all_words = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            clean_text_file(input_path, output_path)
            print(f"Processed {filename}")
            
            with open(output_path, 'r', encoding='utf-8') as file:
                text = file.read()
                words = re.findall(r'\w+', text.lower())
                total_tokens += len(words)
                all_words.extend(words)
    
    unique_words = len(set(all_words))
    print(f"Total tokens across all files: {total_tokens}")
    print(f"Total unique words across all files: {unique_words}")

def main():
    clean_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()