from pathlib import Path    

rd_ = Path(__file__).parents[2]
input_file = rd_ / 'corpus-texts' / 'datasets' / 'everythinglm_corrected.txt'

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

from tiktoken import encoding_for_model
encoding = encoding_for_model("gpt-4o")
total_tokens = sum(len(encoding.encode(line)) for line in lines)
print(f"Total tokens: {total_tokens}")

