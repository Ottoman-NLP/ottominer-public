import sys
import pathlib
import pymupdf4llm
from llama_index.core import (SimpleDirectoryReader,
                               Document
)
import unicodedata

def clean_text(text):
    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # Remove any non-printable characters
    text = ''.join(filter(lambda x: x.isprintable(), text))
    
    return text

def pdf_to_markdown(pdf_path):


    base_output_path = pathlib.Path(r"C:\Users\Administrator\Desktop\cook\Ottoman-NLP\corpus-texts")

    pdf_name = pdf_path.stem

    output_folder = base_output_path / "md_out" / pdf_name

    output_folder.mkdir(parents=True, exist_ok=True)

    # Use only the supported options from pymupdf4llm
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=False,  #TODO: add write_images=True when we have images
        dpi=300,  #TODO: add dpi=300 when we have images
        force_text=True,
        page_chunks=False, #Segmentation is not needed for this task - adjust as needed
        table_strategy="lines_strict",
        fontsize_limit=5,
        extract_words=False,
        show_progress=True
    )

    if isinstance(md_text, list):
        # Process each chunk separately
        cleaned_chunks = []
        for chunk in md_text:
            if 'text' in chunk:
                cleaned_text = clean_text(chunk['text'])
                # Preserve markdown formatting
                if chunk.get('type') == 'header':
                    level = chunk.get('level', 1)
                    cleaned_text = f"{'#' * level} {cleaned_text}"
                cleaned_chunks.append(cleaned_text)
        md_text = '\n\n'.join(cleaned_chunks)
    else:
        md_text = clean_text(md_text)

    output_path = output_folder / "osmanlica.md"

    output_path.write_text(md_text, encoding='utf-8')

    print(f"Markdown file saved as: {output_path}")


    reader = SimpleDirectoryReader(input_files=[str(output_path)])

    documents = reader.load_data()
    
    print(f"Generated {len(documents)} LlamaIndex documents")

    if documents:

        print(f"\nFirst 500 characters of {output_path.name}:")

        print(documents[0].text[:500] + "...")

    return documents

def process_directory(input_dir):

    input_path = pathlib.Path(input_dir)

    for pdf_file in input_path.glob('*.pdf'):

        print(f"\nProcessing: {pdf_file}")
        try:
            pdf_to_markdown(pdf_file)

        except Exception as e:

            print(f"Error processing {pdf_file}: {e}")

            import traceback
            
            traceback.print_exc()

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python pdf_to_markdown.py <path_to_input_directory>")

        sys.exit(1)

    input_directory = sys.argv[1]

    try:
        process_directory(input_directory)

        print(f"\nAll PDF files in {input_directory} have been processed.")

        print("Markdown files are saved in: C:\\Users\\Administrator\\Desktop\\cook\\Ottoman-NLP\\corpus-texts\\md_out")

    except Exception as e:

        print(f"An error occurred: {e}")

        import traceback
        
        traceback.print_exc()
