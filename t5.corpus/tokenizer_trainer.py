import pandas as pd
from tokenizers import Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from pathlib import Path

# Define base directory and paths
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

def tokenize_function(examples):
    inputs = bpe_tokenizer.encode_batch(examples['input_text'])
    targets = bpe_tokenizer.encode_batch(examples['target_text'])

    input_ids = [encoding.ids for encoding in inputs]
    target_ids = [encoding.ids for encoding in targets]

    return {'input_ids': input_ids, 'labels': target_ids}

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

model = T5ForConditionalGeneration.from_pretrained("t5-small")
hf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

data_collator = DataCollatorForSeq2Seq(tokenizer=hf_tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
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