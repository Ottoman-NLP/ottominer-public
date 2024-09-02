from pathlib import Path
import fitz
import glob
from extract_pdf import MostCommonFontExtraction
from applied_function import AppliedRegex

rd = Path(__file__).resolve().parents[2]

save_dir = rd / 'saved'
save_dir.mkdir(parents=True, exist_ok=True)

extracted_text_dir = save_dir / 'extracted_texts'
formatted_text_dir = save_dir / 'formatted_texts'
extracted_text_dir.mkdir(parents=True, exist_ok=True)
formatted_text_dir.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path: str):
    """
    Extracts text from a PDF, processes it for formatting, 
    and saves both the original extracted and formatted versions.

    :param pdf_path: Path to the PDF file.
    """
    task_description = f"Extracting text from {pdf_path}"
    next_task = "Formatting text"

    try:
        with fitz.open(pdf_path) as pdf_document:
            extractor = MostCommonFontExtraction(pdf_document)
            extracted_text = extractor.extract_text(pdf_path)
    except Exception as e:
        print(f"Error while extracting text from PDF: {e}")
        return

    extracted_text_file = extracted_text_dir / f"{Path(pdf_path).stem}_extracted.txt"
    with open(extracted_text_file, 'w', encoding='utf-8') as file:
        file.write(extracted_text)
    
    print(f"Extracted text saved to: {extracted_text_file}")

    formatted_text = format_txt(extracted_text)
    formatted_text = remove_empty_lines(formatted_text)

    formatted_text_file = formatted_text_dir / f"{Path(pdf_path).stem}_formatted.txt"
    with open(formatted_text_file, 'w', encoding='utf-8') as file:
        file.write(formatted_text)
    
    print(f"Formatted text saved to: {formatted_text_file}")

    print(f"Processing completed for: {pdf_path}")

def format_txt(text: str) -> str:
    """
    Formats the text based on regex pattern matching and additional processing.
    
    :param text: Text to be formatted.
    :return: Formatted or modified text.
    """
    try:
        formatter = AppliedRegex(text)
        formatted_text = formatter.apply_format_or_remove()
        formatted_text = formatter.remove_empty_lines(formatted_text)
        formatted_text = formatter._apply_remove(formatted_text)
        formatted_text = formatter.handle_special_line_cases(formatted_text)
        
        print(f"Formatted text length: {len(formatted_text)}")
        return formatted_text
    except Exception as e:
        print(f"Error while formatting text: {e}")
        return text

def remove_empty_lines(text: str) -> str:
    """
    Removes empty lines from the text.
    
    :param text: Text to be cleaned of empty lines.
    :return: Cleaned text.
    """
    cleaned_text = "\n".join([line for line in text.splitlines() if line.strip()])
    print(f"Cleaned text length after removing empty lines: {len(cleaned_text)}")
    return cleaned_text

if __name__ == "__main__":
    pdf_dir = save_dir / 'pdfs'
    pdf_files = glob.glob(str(pdf_dir / '*.pdf'))

    if not pdf_files:
        print("No PDF files found in the directory.")
    else:
        for pdf_file_path in pdf_files:
            print(f"Processing: {pdf_file_path}")
            extract_pdf(pdf_file_path)