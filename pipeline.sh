#!/bin/bash
COLLECTOR_PATH="$(dirname "$(realpath "$0")")/pdf_extractor/collector.py"
ANALYZE_FONTS_PATH="$(dirname "$(realpath "$0")")/pdf_extractor/analyze.py"
TEXT_EXTRACTOR_PATH="$(dirname "$(realpath "$0")")/pdf_extractor/text_extractor.py"
PREPROCESS_TEXT_PATH="$(dirname "$(realpath "$0")")/pdf_extractor/preprocess_text.py"
METRICS_PATH="$(dirname "$(realpath "$0")")/metrics/metrics.py"
PROGRESS_PATH="$(dirname "$(realpath "$0")")/anim/progress.py"
LOG_FILE="$(dirname "$(realpath "$0")")/pipeline_log.txt"

echo "Pipeline started at $(date)" > $LOG_FILE

run_step() {
    local step_name=$1
    local script_path=$2
    echo "Running $step_name..." | tee -a $LOG_FILE
    python3 $PROGRESS_PATH $step_name &
    progress_pid=$!
    start_time=$(date +%s)
    python3 $script_path 2>&1 | tee -a $LOG_FILE
    end_time=$(date +%s)
    kill -SIGINT $progress_pid
    wait $progress_pid 2>/dev/null

    local duration=$(( end_time - start_time ))
    echo "$step_name finished in $duration seconds" | tee -a $LOG_FILE
}

mkdir -p texture/data_pdfs texture/data_txts texture/data_preprocessed_txts metrics

run_step "collector.py" $COLLECTOR_PATH
run_step "analyze.py" $ANALYZE_FONTS_PATH
run_step "text_extractor.py" $TEXT_EXTRACTOR_PATH
run_step "preprocess_text.py" $PREPROCESS_TEXT_PATH
run_step "metrics.py" $METRICS_PATH

echo "Pipeline finished at $(date)" | tee -a $LOG_FILE
