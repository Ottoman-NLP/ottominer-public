import re
from pathlib import Path

def normalize_chars(text):
    char_map = {
        'â': 'a', 'Â': 'A', 'ā': 'a', 'Ā': 'A', 'ê': 'e', 'ē': 'e', 'Ē': 'E',
        'å': 'a', 'Å': 'A', 'î': 'i', 'Î': 'I', 'ī': 'i', 'Ī': 'I',
        'û': 'u', 'ū': 'u', 'Û': 'U', 'ô': 'ö', 'ō': 'o', 'Ō': 'Ö',
        '"': '"', '"': '"', "'": "'", '–': '-', '—': '-', '…': '...'
    }
    for old, new in char_map.items():
        text = text.replace(old, new)
    return text

def clean_text(text):
    text = normalize_chars(text)

    patterns_to_remove = [
        r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]',  # Arabic
        r'[\uE000-\uF8FF]',  # Private Use Area
        r'[()،؛,]',  # Unwanted punctuation
        r'^\s*\d+\.\s+',  # Bullet points
        r'[!.,?;:]{2,}',  # Excessive punctuation
        r'\s{2,}',  # Multiple spaces
        r'^\d+$',  # Stand-alone digits
        r"'[^']*'",  # Remove everything inside single quotes
        r'/[^/]*/',  # Remove everything inside forward slashes
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)
    
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    text = re.sub(r'^([A-ZİĞÜŞÖÇ\s]+)$', r'Başlık: \1', text, flags=re.MULTILINE)
    
    return text.strip()

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def create_clean_test_data():
    project_root = Path(__file__).resolve().parents[2]
    input_file = project_root / "corpus-texts" / "test_data" / "test_data.txt"
    output_file = project_root / "corpus-texts" / "test_data" / "clean_test_data.txt"

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        text = infile.read()
        cleaned_text = clean_text(text)
        sentences = split_into_sentences(cleaned_text)
        for sentence in sentences:
            if sentence.strip():
                outfile.write(sentence.strip() + '\n')

    print(f"Clean test data created successfully: {output_file}")

if __name__ == "__main__":
    print("Creating clean test data...")
    create_clean_test_data()
    print("Process completed.")