import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
from tokenizers import Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from pathlib import Path

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

model = T5ForConditionalGeneration.from_pretrained("t5-small")
hf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize_function(examples):
    model_inputs = hf_tokenizer(examples['input_text'], max_length=128, truncation=True, padding="max_length")
    labels = hf_tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=hf_tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    no_cuda=False,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=hf_tokenizer,
    data_collator=data_collator,
)

trainer.train()
