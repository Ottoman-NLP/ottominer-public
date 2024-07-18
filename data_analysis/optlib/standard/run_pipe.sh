#!/bin/bash

# Define directories and files
RAW_DATA_FILE="$(pwd)/var/raw_test_data.txt"
PROCESSED_DATA_FILE="$(pwd)/var/test_data.txt"
STANDARD_RESULTS_DIR="$(pwd)/var/standard_results"
PROGRESS_SCRIPT="$(pwd)/etc/anim/progress.py"
PROCESS_TEXT_SCRIPT="$(pwd)/optlib/standard/process_text.py"
PROCESS_CORPUS_SCRIPT="$(pwd)/optlib/standard/process_corpus.py"
mkdir -p $STANDARD_RESULTS_DIR
run_with_progress() {
    local task_name=$1
    shift
    local cmd=$@
    python3 $PROGRESS_SCRIPT "$task_name" &
    local progress_pid=$!
    $cmd
    kill $progress_pid
    wait $progress_pid 2>/dev/null
}
echo "Running process_text.py..."
run_with_progress "Processing Text" python3 $PROCESS_TEXT_SCRIPT
echo "Running process_corpus.py..."
run_with_progress "Analyzing Corpus" python3 $PROCESS_CORPUS_SCRIPT
echo "Moving results to $STANDARD_RESULTS_DIR..."
mv $(pwd)/optlib/standard/rank_order_frequency_list.csv $STANDARD_RESULTS_DIR/
mv $(pwd)/optlib/standard/alphabetical_order_frequency_list.csv $STANDARD_RESULTS_DIR/
mv $(pwd)/optlib/standard/concordance_example_term1.txt $STANDARD_RESULTS_DIR/
mv $(pwd)/optlib/standard/concordance_example_term2.txt $STANDARD_RESULTS_DIR/
echo "Pipeline execution completed. Results are in $STANDARD_RESULTS_DIR."