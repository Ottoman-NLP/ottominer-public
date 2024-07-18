import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
anim_dir = os.path.join(current_dir, '../../etc/anim')
progress_path = os.path.join(anim_dir, 'progress.py')
spec = importlib.util.spec_from_file_location("progress", progress_path)
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)
ProgressBar = progress.ProgressBar
from sklearn.metrics import precision_recall_fscore_support


class DataStatistics:
    def __init__(self, data_dir, verbose=False, progress_bar=False):
        self.data_dir = data_dir
        
        self.data = [
            {"name": "siratimustakim", "path": os.path.join(data_dir, "siratimustakim"), "documents": []},
            {"name": "sebiluressad", "path": os.path.join(data_dir, "sebiluressad"), "documents": []},
            {"name": "tarih", "path": os.path.join(data_dir, "tarih"), "documents": []}
        ]
        
        self.corpus_stats = {
            "documents": 0,
            "sentences": 0,
            "tokens": 0,
            "time_periods": {}
        }
        
        self.language_variations = {}
        self.comparison = {}
        self.tables = {}
        self.visualizations = {}
        
        self.verbose = verbose
        self.progress_bar = progress_bar

    def analyze_data(self):
        if self.progress_bar:
            pb = ProgressBar()
            t = pb.start("Analyzing data")
            
        for cilt_number in range(1, 26):  # Assuming there are 25 cilt files
            file_name = f"cilt {cilt_number}.txt"  # Construct file name
            file_path = os.path.join(self.data_dir, file_name)
            
            if not os.path.exists(file_path):  # Handle potential naming inconsistency (space vs underscore)
                file_name = f"cilt_{cilt_number}.txt"  # Try with an underscore
                file_path = os.path.join(self.data_dir, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    sentences = text.split(".")
                    tokens = text.split(" ")
                    
                    self.corpus_stats["documents"] += 1
                    self.corpus_stats["sentences"] += len(sentences)
                    self.corpus_stats["tokens"] += len(tokens)
                    
                    time_period = f"cilt {cilt_number}"  # Use cilt number as the time period
                    if time_period in self.corpus_stats["time_periods"]:
                        self.corpus_stats["time_periods"][time_period] += 1
                    else:
                        self.corpus_stats["time_periods"][time_period] = 1
                    
                    if self.verbose:
                        print(f"Analyzed {file_name}")
            else:
                print(f"File {file_name} not found.")
                        
        if self.progress_bar:
            pb.stop(t)
        
    def evaluate_sentence_segmentation(self, gold_standard_path, test_data_path):
        # Load gold standard sentences
        with open(gold_standard_path, 'r', encoding='utf-8') as file:
            gold_sentences = file.read().split('\n')
        
        # Load test data and segment sentences
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.read()
        segmented_sentences = self.segment_sentences(test_data)  # Assuming this is a method for sentence segmentation
        
        # Calculate TP, FP, FN
        tp = len(set(gold_sentences) & set(segmented_sentences))
        fp = len(set(segmented_sentences) - set(gold_sentences))
        fn = len(set(gold_sentences) - set(segmented_sentences))
        
        # Calculate Precision, Recall, F1 Score
        precision, recall, f1, _ = precision_recall_fscore_support([1]*tp + [0]*fn, [1]*tp + [0]*fp, average='binary')
        
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

    def compare_statistics(self, other_study_stats):
        comparison_results = {}
        for key, value in self.corpus_stats.items():
            if key in other_study_stats:
                difference = value - other_study_stats[key]
                percentage_difference = (difference / other_study_stats[key]) * 100 if other_study_stats[key] else 0
                comparison_results[key] = {
                    'current_study': value,
                    'other_study': other_study_stats[key],
                    'difference': difference,
                    'percentage_difference': f"{percentage_difference:.2f}%"
                }
            else:
                comparison_results[key] = 'Not available in other study'
        
        self.comparison = comparison_results
        for key, result in comparison_results.items():
            print(f"{key}: {result}")

    def create_tables(self):
        self.tables = pd.DataFrame(self.corpus_stats)

    def create_visualizations(self, output_dir):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Bar chart for distribution of documents by time period
        plt.figure(figsize=(10, 5))
        plt.bar(self.corpus_stats["time_periods"].keys(), self.corpus_stats["time_periods"].values(), color='skyblue')
        plt.xlabel('Time Period')
        plt.ylabel('Number of Documents')
        plt.title('Distribution of Documents by Time Period')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distribution_by_time_period.png"))
        plt.close()
        
        # Pie chart for proportion of documents in each time period
        plt.figure(figsize=(8, 8))
        plt.pie(self.corpus_stats["time_periods"].values(), labels=self.corpus_stats["time_periods"].keys(), autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Documents by Time Period')
        plt.savefig(os.path.join(output_dir, "proportion_by_time_period.png"))
        plt.close()
        
        # Histogram for distribution of document lengths
        # Assuming 'document_lengths' is a list of lengths of all documents
        plt.figure(figsize=(10, 5))
        plt.hist(self.corpus_stats["document_lengths"], bins=20, color='purple')
        plt.xlabel('Document Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Document Lengths')
        plt.savefig(os.path.join(output_dir, "distribution_of_document_lengths.png"))
        plt.close()
        
        # Line graph for trend of document publication over time
        # Assuming 'documents_over_time' is a dict with time period as key and number of documents as value
        plt.figure(figsize=(10, 5))
        time_periods_sorted = sorted(self.corpus_stats["time_periods"].items())  # Assuming time periods can be sorted linearly
        plt.plot([item[0] for item in time_periods_sorted], [item[1] for item in time_periods_sorted], marker='o', linestyle='-', color='green')
        plt.xlabel('Time Period')
        plt.ylabel('Number of Documents')
        plt.title('Trend of Document Publication Over Time')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trend_of_document_publication.png"))
        plt.close()

    def display_statistics(self):
        print("Corpus Statistics:")
        print(f"Total number of documents: {self.corpus_stats['documents']}")
        print(f"Total number of sentences: {self.corpus_stats['sentences']}")
        print(f"Total number of tokens: {self.corpus_stats['tokens']}")
        
        print("\nDistribution of documents by time period:")
        for time_period, count in self.corpus_stats["time_periods"].items():
            print(f"{time_period}: {count}")

    def save_statistics(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.create_tables()
        self.tables.to_csv(os.path.join(output_dir, "corpus_stats.csv"), index=False)
        self.create_visualizations(output_dir)

        print(f"Corpus statistics saved to {os.path.join(output_dir, 'corpus_stats.csv')}")
        
if __name__ == "__main__":
    try:
        data_dir = "../../../LLM/texture/txts"
        output_dir = "../../../var/stats/data_evaluation"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except ImportError as e:
        print("Module not found.")
        data_dir = input("Enter the path to the data directory: ")

    ds = DataStatistics(data_dir, verbose=True, progress_bar=True)
    ds.analyze_data()
    ds.evaluate_sentence_segmentation()
    ds.display_statistics()
    ds.save_statistics(output_dir)

    print("Data statistics analysis complete.")