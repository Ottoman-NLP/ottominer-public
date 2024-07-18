#!/bin/bash

# Hard-coded directories
INPUT_DIR="../texture/filtered_txt"
OUTPUT_DIR="../texture/final"

mkdir -p "$OUTPUT_DIR"

echo "Starting the PDF to text extraction process..."
python3 main.py "$INPUT_DIR" "$OUTPUT_DIR"

echo "Starting the text filtering process..."
python3 filter.py "$INPUT_DIR" "$OUTPUT_DIR"

echo "Pipeline completed successfully."