import os
import fitz
from datetime import datetime
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

pdf_dir = "LLM/texture/pdfs/"
txt_dir = "LLM/texture/txts/"
results_dir = "var/results/"

os.makedirs(results_dir, exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../etc/anim')))
from progress import ProgressBar

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = os.path.join(results_dir, f"report_{timestamp}.txt")

def count_words_and_chars(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        chars = len(text)
    return len(words), chars, words

def count_words_in_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        words = text.split()
    return len(words)

def get_all_pdfs(directory):
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def detect_noise(sentences):
    noise_patterns = [
        r'[^\w\s,.?!]',  # non-alphanumeric
        r'^\d+$',        # standalone digits
        r'[^\x00-\x7F]+' # non-ASCII
    ]
    noise_counts = {pattern: 0 for pattern in noise_patterns}
    for sentence in sentences:
        for pattern in noise_patterns:
            if re.search(pattern, sentence):
                noise_counts[pattern] += 1
    return noise_counts

import matplotlib.pyplot as plt

def plot_noise_data_combined(report_summary, threshold, output_path):
    patterns = list(report_summary[0]['noise_counts'].keys())
    num_patterns = len(patterns)
    num_files = len(report_summary)
    
    noise_data = {pattern: [] for pattern in patterns}
    file_names = []

    for entry in report_summary:
        if entry['noise_counts'] != "N/A":
            file_name = entry['file']
            file_names.append(file_name)
            noise_counts = entry['noise_counts']
            for pattern in patterns:
                noise_data[pattern].append(noise_counts[pattern])

    fig, ax = plt.subplots(figsize=(12, 8))

    for pattern in patterns:
        data_points = noise_data[pattern]
        ax.plot(file_names, data_points, marker='o', label=f'Noise Pattern: {pattern}')

    # Plot mean and standard deviation
    for pattern in patterns:
        data_points = noise_data[pattern]
        mean_val = np.mean(data_points)
        std_dev = np.std(data_points)
        ax.axhline(y=mean_val, color='blue', linestyle='-.', label=f'Mean {pattern}')
        ax.axhline(y=mean_val + std_dev, color='red', linestyle='--', label=f'+1 Std Dev {pattern}')
        ax.axhline(y=mean_val - std_dev, color='orange', linestyle='--', label=f'-1 Std Dev {pattern}')

    ax.axhline(y=threshold, color='red', linestyle='-', label=f'Threshold: {threshold}')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Files')
    plt.ylabel('Counts')
    plt.title('Noise Pattern Counts for All Files')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

pdf_files = get_all_pdfs(pdf_dir)
txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

num_pdfs = len(pdf_files)
num_txts = len(txt_files)

pb = ProgressBar()
t = pb.start("Processing files")

with open(report_file, 'w', encoding='utf-8') as report:
    report.write(f"Number of PDF files: {num_pdfs}\n")
    report.write(f"Number of TXT files: {num_txts}\n\n")

    report_summary = []

    for i, pdf_path in enumerate(pdf_files):
        pdf_file = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_file)[0]
        txt_file = base_name + ".txt"

        txt_path = os.path.join(txt_dir, txt_file)

        if txt_file in txt_files:
            pdf_word_count = count_words_in_pdf(pdf_path)
            txt_word_count, txt_char_count, txt_words = count_words_and_chars(txt_path)

            sentences = re.split(r'[.!?]', open(txt_path, 'r', encoding='utf-8').read())
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            num_sentences = len(sentences)
            words_per_sentence = [len(sentence.split()) for sentence in sentences]
            noise_counts = detect_noise(sentences)

            report_summary.append({
                "file": pdf_file,
                "pdf_words": pdf_word_count,
                "txt_words": txt_word_count,
                "txt_chars": txt_char_count,
                "avg_words_per_sentence": np.mean(words_per_sentence),
                "std_dev_words_per_sentence": np.std(words_per_sentence),
                "num_sentences": num_sentences,
                "noise_counts": noise_counts
            })
        else:
            report_summary.append({
                "file": pdf_file,
                "pdf_words": "N/A",
                "txt_words": "File missing",
                "txt_chars": "File missing",
                "avg_words_per_sentence": "N/A",
                "std_dev_words_per_sentence": "N/A",
                "num_sentences": "N/A",
                "noise_counts": "N/A"
            })

        pb.update("Processing files", int((i + 1) / num_pdfs * 100))

    pb.stop(t)

    for entry in report_summary:
        report.write(f"File: {entry['file']}\n")
        report.write(f"  PDF Words: {entry['pdf_words']}\n")
        report.write(f"  TXT Words: {entry['txt_words']}\n")
        report.write(f"  TXT Chars: {entry['txt_chars']}\n")
        report.write(f"  Avg Words per Sentence: {entry['avg_words_per_sentence']}\n")
        report.write(f"  Std Dev Words per Sentence: {entry['std_dev_words_per_sentence']}\n")
        report.write(f"  Number of Sentences: {entry['num_sentences']}\n")
        report.write(f"  Noise Counts: {entry['noise_counts']}\n")
        report.write("\n")

    total_pdfs = len(report_summary)
    total_txts = sum(1 for entry in report_summary if entry['txt_words'] != "File missing")

    if total_txts > 0:
        avg_txt_words = sum(entry['txt_words'] for entry in report_summary if entry['txt_words'] != "File missing") / total_txts
        avg_txt_chars = sum(entry['txt_chars'] for entry in report_summary if entry['txt_chars'] != "File missing") / total_txts
    else:
        avg_txt_words = avg_txt_chars = 0

    report.write(f"Total PDFs processed: {total_pdfs}\n")
    report.write(f"Total TXTs available: {total_txts}\n")
    report.write(f"Average TXT word count: {avg_txt_words}\n")
    report.write(f"Average TXT character count: {avg_txt_chars}\n")

noise_plot_combined_path = os.path.join(results_dir, f"noise_plot_combined_{timestamp}.png")
plot_noise_data_combined(report_summary, threshold=5000, output_path=noise_plot_combined_path)