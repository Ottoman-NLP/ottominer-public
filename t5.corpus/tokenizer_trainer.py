import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
from tokenizers import Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Config
from datasets import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

base_dir = Path(__file__).resolve().parents[2]
input_path = base_dir / 'corpus-texts' / 'tokenizer_bpe' / 'bpe_tokenizer.json'
csv_path = base_dir / 'corpus-texts' / 'automated.csv.results' / 'all_aligned_sentences.csv'
output_dir = base_dir / 'corpus-texts' / 'tokenizer_bpe'

bpe_tokenizer = Tokenizer.from_file(str(input_path))
print("Tokenizer loaded successfully.")

try:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print(f"CSV loaded successfully with {len(df)} rows.")
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)

if 'noisy' not in df.columns or 'clean' not in df.columns:
    print("Error: Required columns ('noisy', 'clean') not found in the CSV.")
    exit(1)

data_dict = {
    'input_text': df['noisy'].tolist(),
    'target_text': df['clean'].tolist()
}

dataset = Dataset.from_dict(data_dict)

dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
test_dataset = dataset['test']
vocab_size = bpe_tokenizer.get_vocab_size()
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", vocab_size=vocab_size)
t5_tokenizer.add_tokens(list(bpe_tokenizer.get_vocab().keys()))
config = T5Config.from_pretrained("t5-small", vocab_size=len(t5_tokenizer))
model = T5ForConditionalGeneration(config)

@dataclass
class CustomDataCollator:
    tokenizer: T5Tokenizer
    model: Any
    padding: bool = True
    max_length: Union[int, None] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]
       
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def tokenize_function(examples):
    inputs = t5_tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=512)
    targets = t5_tokenizer(examples['target_text'], truncation=True, padding='max_length', max_length=512)

    vocab_size = len(t5_tokenizer)
    inputs['input_ids'] = [[min(token, vocab_size - 1) for token in sequence] for sequence in inputs['input_ids']]
    targets['input_ids'] = [[min(token, vocab_size - 1) for token in sequence] for sequence in targets['input_ids']]

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }

print(f"T5Tokenizer vocabulary size: {len(t5_tokenizer)}")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

data_collator = CustomDataCollator(tokenizer=t5_tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    no_cuda=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=t5_tokenizer,
    data_collator=data_collator,
)

trainer.train()