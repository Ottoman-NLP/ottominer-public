from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parents[2]

input_path = base_dir / 'corpus-texts' / 'automated.csv.results' / 'all_aligned_sentences.csv'
output_dir = base_dir / 'corpus-texts' / 'tokenizer_bpe'

output_dir.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(input_path, on_bad_lines='skip')  # Skip problematic lines
    print(f"CSV loaded successfully with {len(df)} rows.")
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)

# Choose the column(s) for training the tokenizer
# Combine 'noisy' and 'clean' text if desired
texts = df['noisy'].tolist() + df['clean'].tolist()  # Use both columns

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
)

tokenizer.train_from_iterator(texts, trainer)
output_file = output_dir / 'bpe_tokenizer.json'
tokenizer.save(str(output_file))

print(f"BPE tokenizer trained and saved at '{output_file}'")