import os
from pathlib import Path
import re
from tiktoken import encoding_for_model

def count_tokens(text):
    encoding = encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def analyze_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    token_count = count_tokens(content)
    sentence_count = len(re.findall(r'\w+[.!?](?:\s|$)', content))
    word_count = len(content.split())
    char_count = len(content)
    
    return {
        'tokens': token_count,
        'sentences': sentence_count,
        'words': word_count,
        'characters': char_count
    }

def process_folders(root_dir):
    total_stats = {
        'tokens': 0,
        'sentences': 0,
        'words': 0,
        'characters': 0,
        'files_processed': 0
    }

    for folder_path in Path(root_dir).iterdir():
        if folder_path.is_dir():
            osmanlica_file = folder_path / "osmanlıca.txt"
            if osmanlica_file.exists():
                print(f"Processing: {osmanlica_file}")
                stats = analyze_file(osmanlica_file)
                
                info_file = folder_path / "info.txt"
                with open(info_file, 'w', encoding='utf-8') as f:
                    f.write(f"File: {osmanlica_file.name}\n")
                    f.write(f"Token count: {stats['tokens']}\n")
                    f.write(f"Sentence count: {stats['sentences']}\n")
                    f.write(f"Word count: {stats['words']}\n")
                    f.write(f"Character count: {stats['characters']}\n")
                
                print(f"Info file created: {info_file}")

                # Update total stats
                for key in ['tokens', 'sentences', 'words', 'characters']:
                    total_stats[key] += stats[key]
                total_stats['files_processed'] += 1
            else:
                print(f"No osmanlıca.txt found in {folder_path}")

    return total_stats

def main():
    rd_ = Path(__file__).parents[2]
    texts_dir = rd_ / 'corpus-texts' / 'texts'
    
    print(f"Scanning directory: {texts_dir}")
    total_stats = process_folders(texts_dir)
    
    # Create total info file
    total_info_file = texts_dir / "total_info.txt"
    with open(total_info_file, 'w', encoding='utf-8') as f:
        f.write("Total Statistics for all osmanlıca.txt files\n")
        f.write("===========================================\n")
        f.write(f"Files processed: {total_stats['files_processed']}\n")
        f.write(f"Total token count: {total_stats['tokens']}\n")
        f.write(f"Total sentence count: {total_stats['sentences']}\n")
        f.write(f"Total word count: {total_stats['words']}\n")
        f.write(f"Total character count: {total_stats['characters']}\n")
    
    print(f"Total info file created: {total_info_file}")
    print("Processing complete.")

if __name__ == "__main__":
    main()