import fitz
import os


def normalize_characters(text):
    replacements = {
        "“": '"', "”": '"', "’": "'", "–": "-", "â": "a", "ê": "e", "î": "i",
        "ô": "ö", "û": "ü", "ī": "i", "ū": "ü", "ō": "o", "ē": "e", "ā": "a",
        "Â": "A", "Ê": "E", "Î": "İ", "Ô": "Ö", "Û": "Ü", "Ī": "İ", "Ū": "Ü",
        "Ō": "Ö", "Ē": "E", "Ā": "A", "": "", "": "", "œ": "i",
        "†": "u", "å": "a", "¢": "i", "": "", "": ""
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_text(pdf_path, txt_output_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open PDF: {pdf_path}. Error: {e}")
        return

    text_output = []

    for page in doc:
        text = page.get_text("text")
        text = normalize_characters(text)

        lines = text.split('\n')
        processed_text = []

        for line in lines:
            line = line.strip()
            if line.isdigit() or not line:
                continue
            if line.startswith("====") or line.endswith("===="):
                processed_text.append("----{}----".format(line.replace("====", "").strip()))
            else:
                if not line.startswith(("SEBİLÜRREŞAD", "CİLD", "ADED", "SAYFA")):
                    processed_text.append(line)

        text_output.append("\n".join(processed_text))

    doc.close()

    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(text_output))

    print(f"Processed and wrote to {txt_output_path}")


def process_directory(directory_path, output_directory):
    specific_order = ["cilt_25.pdf", "cilt_24.pdf", "cilt_23.pdf", "cilt_22.pdf",
                      "cilt_20.pdf", "cilt_19.pdf", "cilt_17.pdf", "cilt_16.pdf", "cilt_14.pdf"]

    files_to_process = []

    for root, dirs, files in os.walk(directory_path):
        for file_name in specific_order:
            if file_name in files:
                pdf_path = os.path.join(root, file_name)
                txt_path = os.path.join(output_directory, file_name.replace('.pdf', '.txt'))
                files_to_process.append((pdf_path, txt_path))


    for pdf_path, txt_path in files_to_process:
        print(f"Starting processing PDF: {pdf_path}")
        extract_text(pdf_path, txt_path)


pdf_directory_path = '../texture/data_pdfs/sebiluressad'
output_txt_directory = '../texture/data_txt/sebiluressad'
os.makedirs(output_txt_directory, exist_ok=True)
process_directory(pdf_directory_path, output_txt_directory)

