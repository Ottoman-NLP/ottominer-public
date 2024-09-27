from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
import torch
from pathlib import Path

rd_ = Path(__file__).resolve().parents[2]
at5_ = rd_ / 'corpus-texts' / 'apply_t5'
at5_original = at5_ / 'original.txt'
at5_corrected = at5_ / 'corrected.txt'

model_path = rd_ / 'corpus-texts' / 'trained_model' / 'final_model'
model = T5ForConditionalGeneration.from_pretrained(str(model_path))
tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA device not available, using CPU.")

model.to(device)

def format_sentence(sentence):
    inputs = tokenizer("format: " + sentence, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs.pop('token_type_ids', None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


at5_.mkdir(parents=True, exist_ok=True)

with open(at5_original, 'r', encoding='utf-8') as infile, open(at5_corrected, 'w', encoding='utf-8') as outfile:
    content = infile.read()

    chunks = content.split()
    chunk_size = 100
    for i in range(0, len(chunks), chunk_size):
        chunk = ' '.join(chunks[i:i+chunk_size])
        formatted_chunk = format_sentence(chunk)
        

        outfile.write(f"Input: {chunk}\n")
        outfile.write(f"Output: {formatted_chunk}\n\n")

        print(f"Input: {chunk}")
        print(f"Output: {formatted_chunk}")
        print()

print(f"Processing complete. Results written to {at5_corrected}")