import sys
import pathlib
import pymupdf4llm
from llama_index.core import (
    SimpleDirectoryReader, 
    Document
)

"""
In order to use this script, please use your local python compiler as follows:
< python.exe pdf_to_markdown.py <path_to_input_directory> > 

For further information, please refer to /ottominer/mine/wst.md
"""

def pdf_to_markdown(pdf_path, output_folder, write_images=True, dpi=150):
    """pdf to markdown generation"""
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=write_images,
        dpi=dpi,
        page_chunks=False
    )

    output_path = pathlib.Path(output_folder) / f"{pathlib.Path(pdf_path).stem}.md"
    output_path.write_text(md_text, encoding='utf-8')
    print(f"Markdown file saved as: {output_path}")
    reader = SimpleDirectoryReader(input_files=[str(output_path)])
    documents = reader.load_data()
    
    print(f"Generated {len(documents)} LlamaIndex documents")

    if documents:
        print(f"\nFirst 500 characters of {output_path.name}:")
        print(documents[0].text[:500] + "...")

    return documents

def process_directory(input_dir, output_dir):
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)


    output_path.mkdir(parents=True, exist_ok=True)


    for pdf_file in input_path.glob('*.pdf'):
        print(f"\nProcessing: {pdf_file}")
        try:
            pdf_to_markdown(pdf_file, output_path)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_markdown.py <path_to_input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = "markdown_output"

    try:
        process_directory(input_directory, output_directory)
        print(f"\nAll PDF files in {input_directory} have been processed.")
        print(f"Markdown files are saved in: {output_directory}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
