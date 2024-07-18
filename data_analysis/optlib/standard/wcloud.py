import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
lib_dir = os.path.join(project_root, 'lib')
sys.path.append(lib_dir)

from modules import *

nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

class TextAnalysis:
    
    def __init__(self, main_data_path: str="LLM/texture/txts", test_data_path: str="var/test_data.txt", tmp_dir: str="tmp"):
        self.main_data_path = os.path.abspath(main_data_path)
        self.test_data_path = os.path.abspath(test_data_path)
        self.tmp_dir = os.path.abspath(tmp_dir)
        self.test_data = self.load_test_data()
        self.clean_test_data = self.clean_text(self.test_data)
        self.main_data_samples = self.load_and_sample_main_data()
        self.clean_main_data_samples = self.clean_text(self.main_data_samples)
        
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    
    def load_test_data(self):
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = " ".join(text.strip() for text in text.split("\n"))
        return lines
    
    def load_and_sample_main_data(self, num_samples=5):
        sampled_lines = []
        for i in range(1, 26):  # Assuming file names are cilt_1.txt to cilt_25.txt
            file_path = os.path.join(self.main_data_path, f"cilt_{i}.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                sampled_lines.extend(random.sample(lines, min(num_samples, len(lines))))
        sampled_text = " ".join(sampled_lines)
        return sampled_text
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text
    
    def analyze_text(self, text):
        freq_dist = nltk.FreqDist(text.split())
        return freq_dist

    def generate_wordcloud(self, text, title):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(os.path.join(self.tmp_dir, f"{title.replace(' ', '_')}.png"))
        plt.close()
    
    def ngram_analysis(self, text, n=2):
        tokens = text.split()
        ngrams = list(nltk.ngrams(tokens, n))
        ngram_freq_dist = nltk.FreqDist(ngrams)
        return ngram_freq_dist.most_common(20)
    
    def ner_and_pos_tagging(self, text):
        doc = nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        pos_tags = [(token.text, token.pos_) for token in doc]
        return entities, pos_tags

    def save_results(self, results, file_prefix):
        for result_type, data in results.items():
            file_path = os.path.join(self.tmp_dir, f"{file_prefix}_{result_type}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                if isinstance(data, list):
                    for item in data:
                        f.write(f"{item}\n")
                else:
                    f.write(str(data))

# Example usage
if __name__ == "__main__":
    # Dynamically add the lib directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    lib_dir = os.path.join(project_root, 'lib')
    sys.path.append(lib_dir)

    analysis = TextAnalysis()
    
    # Analyze main data samples
    main_freq_dist = analysis.analyze_text(analysis.clean_main_data_samples)
    main_common_words = main_freq_dist.most_common(20)
    
    # Analyze test data
    test_freq_dist = analysis.analyze_text(analysis.clean_test_data)
    test_common_words = test_freq_dist.most_common(20)
    
    # Generate word clouds
    analysis.generate_wordcloud(analysis.clean_main_data_samples, "Main Data Word Cloud")
    analysis.generate_wordcloud(analysis.clean_test_data, "Test Data Word Cloud")
    
    # N-gram analysis
    main_common_bigrams = analysis.ngram_analysis(analysis.clean_main_data_samples, n=2)
    test_common_bigrams = analysis.ngram_analysis(analysis.clean_test_data, n=2)
    
    # NER and POS tagging
    main_entities, main_pos_tags = analysis.ner_and_pos_tagging(analysis.clean_main_data_samples)
    test_entities, test_pos_tags = analysis.ner_and_pos_tagging(analysis.clean_test_data)
    
    # Collect results
    results = {
        "main_data_common_words": main_common_words,
        "main_data_common_bigrams": main_common_bigrams,
        "main_data_named_entities": main_entities,
        "main_data_pos_tags": main_pos_tags,
        "test_data_common_words": test_common_words,
        "test_data_common_bigrams": test_common_bigrams,
        "test_data_named_entities": test_entities,
        "test_data_pos_tags": test_pos_tags
    }
    
    # Save results to separate files
    analysis.save_results(results, "analysis_results")

    # Print results to console
    for result_type, data in results.items():
        print(f"{result_type}: {data}")
