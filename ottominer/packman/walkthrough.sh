#!/bin/zsh

dir_exist() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        if [ $? -eq 0 ]; then
            echo "Directory created: $1"
        else
            echo "Error: Failed to create directory: $1"
            exit 1
        fi
    fi
}

# First, read the TOP_DOC_DIR value from the config file
TOP_DOC_DIR=$(sed -n 's/^TOP_DOC_DIR *=//p' config.ini | sed 's/^ *//; s/ *$//')

# Then, ensure the directory exists
dir_exist "$TOP_DOC_DIR"

# Rest of your script remains the same
echo "Total PDF files found:"
find "$TOP_DOC_DIR" -type f -name "*.pdf" | wc -l

echo "\nProcessing PDF files:"

find "$TOP_DOC_DIR" -type f -name "*.pdf" -print0 | while IFS= read -r -d '' file; do
    echo "\nProcessing: $file"
    sleep 0.3
    echo "File size: $(du -h "$file" | cut -f1)"
    # Uncomment the next line if pdfinfo is installed
    # echo "Page count: $(pdfinfo "$file" 2>/dev/null | grep "Pages:" | awk '{print $2}')"
    # Replace ./count.sh with the actual command you want to run on each PDF
    # ./count.sh "$file"
done

if [ $? -eq 0 ]; then
    echo "\nAll PDF files have been processed successfully."
else
    echo "\nError: Processing PDF files failed."
fi
