import os
import sys
import random
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class DataStatistics:
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
    
    def segment_words(self, text):
        return word_tokenize(text)

    def create_gold_standard(self, sample_size=10):
        files = [file for file in os.listdir(self.data_dir) if file.endswith('.txt')]
        sample_files = random.sample(files, min(sample_size, len(files)))
        gold_standard_path = os.path.join(self.data_dir, "gold_standard")
        if not os.path.exists(gold_standard_path):
            os.makedirs(gold_standard_path)
        for file in sample_files:
            output_file_path = os.path.join(gold_standard_path, file)
            if not os.path.exists(output_file_path):
                with open(os.path.join(self.data_dir, file), "r", encoding="utf-8") as f:
                    lines = [next(f) for _ in range(100)]
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
        print(f"Gold standard data created at: {gold_standard_path}")

    def load_gold_standard(self):
        gold_standard_path = os.path.join(self.data_dir, "gold_standard", "cilt_1.txt")
        words = []
        try:
            with open(gold_standard_path, "r", encoding="utf-8") as f:
                text = f.read()
                words.extend(word_tokenize(text))
            print(f"Loaded gold standard data from: {gold_standard_path}")
        except FileNotFoundError:
            print(f"File not found: {gold_standard_path}")
        except Exception as e:
            print(f"Error reading file {gold_standard_path}: {e}")
        return [word.strip() for word in words if word.strip()]
    
    def load_test_data(self):
        test_data_path = os.path.join(self.data_dir, "gold_standard", "cilt_1.txt")
        text = ""
        try:
            with open(test_data_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Loaded test data from: {test_data_path}")
        except FileNotFoundError:
            print(f"File not found: {test_data_path}")
        except Exception as e:
            print(f"Error reading file {test_data_path}: {e}")
        return text
    
    def evaluate_word_segmentation(self):
        gold_standard_words = self.load_gold_standard()
        test_data = self.load_test_data()
        
        if not gold_standard_words:
            print("No gold standard words found.")
            return
        
        if not test_data:
            print("No test data found.")
            return

        # Tokenize the test data
        test_data_clean = test_data.replace('\n', ' ').strip()
        algorithm_output = self.segment_words(test_data_clean)
        
        # Adjust algorithm_output to match the gold standard format
        algorithm_output = [word.strip() for word in algorithm_output]
        
        # Debug output
        print("\nGold Standard Words:")
        for word in gold_standard_words:
            print(f"'{word}'")
        
        print("\nAlgorithm Output Words:")
        for word in algorithm_output:
            print(f"'{word}'")

        correct = sum([1 for gs, ao in zip(gold_standard_words, algorithm_output) if gs == ao])
        total = len(gold_standard_words)
        
        accuracy = correct / total if total != 0 else 0

        print(f"\nWord Segmentation Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # Dynamically add the lib directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    lib_dir = os.path.join(project_root, 'lib')
    sys.path.append(lib_dir)

    data_dir = "/absolute/path/to/LLM/texture/txts"  # Adjust this path to your actual data directory

    ds = DataStatistics(data_dir)
    ds.create_gold_standard(sample_size=1)  # Assuming cilt_1.txt is part of this sample
    ds.evaluate_word_segmentation()