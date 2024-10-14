import os
from pathlib import Path
import warnings
import time
from tqdm import tqdm   
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

def calculate_cleanliness(original, corrected):
    return sum(a == b for a, b in zip(original, corrected)) / max(len(original), len(corrected))

def create_llm():
    return OllamaLLM(
        model="aya:latest",
        callbacks=[StreamingStdOutCallbackHandler()],
        stop=["Human:", "Assistant:", "\n\n"],
        temperature=0.1,
        top_p=0.1,
        repeat_penalty=1.2,
    )

def create_prompt():
    template = """Görev: Aşağıdaki Osmanlı Türkçesi cümleyi düzelt ve temizle. Cümleyi anlamlı ve doğru bir şekilde yeniden düzenle. Osmanlı Türkçesi gramerini ve cümle anlamını koru. Noktalama işaretlerini düzelt ve gerektiğinde ekle. Kelimeleri doğru şekilde ayır ve yazım hatalarını düzelt. Sadece düzeltilmiş cümleyi ver. Başka hiçbir şey ekleme.

Cümle:
{text}

Düzeltilmiş cümle:"""
    return PromptTemplate(template=template, input_variables=["text"])

def post_process(original, corrected):
    # Existing post-processing
    corrected = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s\d.,;:!?()-]', '', corrected)
    corrected = re.sub(r'([.,;:!?](?=\S))', r'\1 ', corrected)
    corrected = re.sub(r'\s+', ' ', corrected).strip()
    corrected = corrected.replace('›', 'ı').replace('‹', 'i')
    
    # Additional post-processing for abbreviations and Sultan titles
    corrected = re.sub(r'(Hz)\.\s+', r'\1. ', corrected)  # Correct spacing after Hz.
    corrected = re.sub(r'Sultan\s+([IVX]+)\.\s*(\w+)', r'Sultan \1. \2', corrected)  # Correct Sultan numeral formatting
    
    return corrected

def needs_correction(sentence):
    # Check for severe issues that indicate a need for correction
    severe_issues = [
        r'\d+\s*\]\s*,\s*\d+',  # Numbers with brackets and commas (e.g., "1037], 5")
        r'\w+\d+\w+',           # Words with numbers in the middle (e.g., "kaddesallahü5")
        r'[a-zA-Z]{20,}',       # Very long words (likely concatenated)
        r'(\w+)(\1\s*){3,}',    # Words repeated more than 3 times
        r'\([^)]*\d+[^)]*\)',   # Parentheses containing numbers
        r'\w+\s*-\s*\w+\s*-\s*\w+',  # Multiple words connected with hyphens
        r'\b(\w{1,2}\s+){3,}',  # Sequence of very short words (1-2 characters)
        r'\b[A-Z][a-z]*\.\s+[A-Z][a-z]*\.',  # Incorrect abbreviation spacing (e.g., "Hz. ")
        r'Sultan\s+[IVX]+\.\s*\n',  # Incorrect Sultan numeral formatting
        r'\s+[.,;:!?]',         # Spaces before punctuation
        r'[.,;:!?](?!\s|$)',    # Missing spaces after punctuation
    ]
    return any(re.search(pattern, sentence) for pattern in severe_issues)

def process_sentence(sentence, chain):
    if needs_correction(sentence):
        try:
            corrected_sentence = chain.run(text=sentence)
            corrected_sentence = post_process(sentence, corrected_sentence)
            cleanliness = calculate_cleanliness(sentence, corrected_sentence)
            return corrected_sentence, cleanliness, True
        except Exception as e:
            print(f"Error processing sentence: {sentence[:50]}...")
            print(f"Error message: {str(e)}")
            return sentence, 0, False
    else:
        return sentence, 1, False

def process_file(input_file, output_file, stats_file):
    llm = create_llm()
    prompt = create_prompt()
    chain = LLMChain(llm=llm, prompt=prompt)

    cleanliness_scores = []
    original_texts = []
    corrected_texts = []
    sentences_processed = 0
    sentences_corrected = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(stats_file, 'w', encoding='utf-8') as statsfile:
        
        for line in tqdm(infile, desc="Processing lines"):
            sentences = re.split(r'(?<=[.!?])\s+', line.strip())
            for sentence in sentences:
                sentences_processed += 1
                corrected_sentence, cleanliness, was_corrected = process_sentence(sentence, chain)
                
                outfile.write(corrected_sentence + ' ')
                if was_corrected:
                    sentences_corrected += 1
                    cleanliness_scores.append(cleanliness)
                    statsfile.write(f"Original: {sentence}\nCorrected: {corrected_sentence}\nCleanliness: {cleanliness:.2f}\n\n")
                    original_texts.append(sentence)
                    corrected_texts.append(corrected_sentence)
            
            outfile.write('\n')

    print(f"Total sentences processed: {sentences_processed}")
    print(f"Sentences corrected: {sentences_corrected}")
    return original_texts, corrected_texts, cleanliness_scores

def analyze_results(original_texts, corrected_texts, cleanliness_scores):
    original_tokens = [word for text in original_texts for word in text.split()]
    corrected_tokens = [word for text in corrected_texts for word in text.split()]
    original_chars = ''.join(original_texts)
    corrected_chars = ''.join(corrected_texts)
    original_sentences = [sent for text in original_texts for sent in re.split(r'[.!?]+', text) if sent]
    corrected_sentences = [sent for text in corrected_texts for sent in re.split(r'[.!?]+', text) if sent]

    plt.figure(figsize=(20, 15))
    plt.subplot(2, 2, 1)
    plt.bar(['Original', 'Corrected'], [len(original_tokens), len(corrected_tokens)])
    plt.title('Token Count Comparison')
    plt.ylabel('Number of Tokens')
    plt.subplot(2, 2, 2)
    plt.bar(['Original', 'Corrected'], [len(original_chars), len(corrected_chars)])
    plt.title('Character Count Comparison')
    plt.ylabel('Number of Characters')
    plt.subplot(2, 2, 3)
    sns.histplot(data=[len(sent.split()) for sent in original_sentences], kde=True, label='Original')
    sns.histplot(data=[len(sent.split()) for sent in corrected_sentences], kde=True, label='Corrected')
    plt.title('Sentence Length Distribution')
    plt.xlabel('Words per Sentence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.subplot(2, 2, 4)
    sns.histplot(data=cleanliness_scores, kde=True)
    plt.title('Cleanliness Score Distribution')
    plt.xlabel('Cleanliness Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('text_cleaning_analysis.png')
    plt.close()

    report = f"""
    Text Cleaning Analysis Report
    =============================
    
    Token Count:
    - Original: {len(original_tokens)}
    - Corrected: {len(corrected_tokens)}
    - Difference: {len(original_tokens) - len(corrected_tokens)}
    
    Character Count:
    - Original: {len(original_chars)}
    - Corrected: {len(corrected_chars)}
    - Difference: {len(original_chars) - len(corrected_chars)}
    
    Sentence Count:
    - Original: {len(original_sentences)}
    - Corrected: {len(corrected_sentences)}
    - Difference: {len(original_sentences) - len(corrected_sentences)}
    
    Average Sentence Length (in words):
    - Original: {sum(len(sent.split()) for sent in original_sentences) / len(original_sentences):.2f}
    - Corrected: {sum(len(sent.split()) for sent in corrected_sentences) / len(corrected_sentences):.2f}
    
    Cleanliness Scores:
    - Average: {sum(cleanliness_scores) / len(cleanliness_scores):.2f}
    - Minimum: {min(cleanliness_scores):.2f}
    - Maximum: {max(cleanliness_scores):.2f}
    
    A detailed visualization has been saved as 'text_cleaning_analysis.png'.
    """
    
    return report

def main():
    rd_ = Path(__file__).parents[2]
    input_file = rd_ / 'corpus-texts' / 'datasets' / 'everythinglm_corrected.txt'
    output_file = rd_ / 'corpus-texts' / 'datasets' / 'everyfile.txt'
    stats_file = rd_ / 'corpus-texts' / 'datasets' / 'correction_stats.txt'
    os.makedirs(output_file.parent, exist_ok=True)

    print("Starting text correction process...")
    start_time = time.time()
    original_texts, corrected_texts, cleanliness_scores = process_file(input_file, output_file, stats_file)
    end_time = time.time()

    print(f"Process completed in {end_time - start_time:.2f} seconds.")
    print(f"Corrected text written to {output_file}")
    print(f"Correction statistics written to {stats_file}")

    if original_texts and corrected_texts and cleanliness_scores:
        print("\nGenerating analysis report...")
        report = analyze_results(original_texts, corrected_texts, cleanliness_scores)
        print(report)

        with open('text_cleaning_analysis_report.txt', 'w', encoding='utf-8') as report_file:
            report_file.write(report)
        print("Analysis report saved to 'text_cleaning_analysis_report.txt'")
        print("Analysis visualization saved to 'text_cleaning_analysis.png'")
    else:
        print("\nSkipping analysis due to lack of processed data.")

if __name__ == "__main__":
    main()
