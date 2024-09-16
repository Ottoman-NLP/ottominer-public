import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import json
import pandas as pd
from tokenizers import Tokenizer
from transformers import T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from pathlib import Path
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

base_dir = Path(__file__).resolve().parents[2]
bpe_tokenizer_path = base_dir / 'corpus-texts' / 'tokenizer_bpe' / 'bpe_tokenizer.json'
sp_model_path = base_dir / 'corpus-texts' / 'ottoman_sp.model'
csv_path = base_dir / 'corpus-texts' / 'automated.csv.results' / 'all_aligned_sentences.csv'
test_data_path = base_dir / 'corpus-texts' / 'test_data' / 'test_data.json'
output_dir = base_dir / 'corpus-texts' / 'trained_model'
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(bpe_tokenizer_path))
tokenizer.pad_token = tokenizer.eos_token
print("Custom BPE tokenizer loaded successfully.")

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
val_dataset = dataset['test']

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_dataset = Dataset.from_dict({
    'input_text': [item['noisy'] for item in test_data],
    'target_text': [item['clean'] for item in test_data]
})

model = T5ForConditionalGeneration.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to: {device}")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    fp16=torch.cuda.is_available(),
    logging_dir=str(output_dir / "logs"),
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting model training...")
trainer.train()

model.save_pretrained(output_dir / "final_model")
tokenizer.save_pretrained(output_dir / "final_model")
print(f"Model and tokenizer saved to {output_dir / 'final_model'}")


print("Evaluating the model on validation set...")
eval_results = trainer.evaluate()
print(eval_results)

print("Evaluating the model on test set...")
test_results = trainer.evaluate(tokenized_test)
print(test_results)

def format_sentence(sentence):
    inputs = tokenizer("format: " + sentence, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nTesting the model with example sentences from the test set:")
for i in range(min(5, len(test_data))):
    input_sentence = test_data[i]['noisy']
    expected_output = test_data[i]['clean']
    formatted = format_sentence(input_sentence)
    print(f"Input: {input_sentence}")
    print(f"Expected: {expected_output}")
    print(f"Model output: {formatted}")
    print()