import fitz  # PyMuPDF

doc = fitz.open("input.pdf")
page = doc.load_page(0)
raw_dict = page.get_text("rawdict")

# Inspect the raw dictionary structure
import pprint
pprint.pprint(raw_dict)
