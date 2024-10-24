import fitz  # PyMuPDF
import pdfminer.encodingdb
from pdfminer.pdfinterp import adobe_glyph_list

def get_char_map(doc, xref):
    font_info = doc.extract_font(xref)
    char_map = {}

    # Try to use cmap
    cmap = font_info.get("cmap")
    if cmap:
        return cmap

    # Try to use glyph names
    glyphs = font_info.get("glyphs")
    if glyphs:
        for ccode, glyph_name in glyphs.items():
            unicode_char = adobe_glyph_list.get(glyph_name)
            if unicode_char:
                char_map[ccode] = ord(unicode_char)
        return char_map

    # Try to use standard encoding
    encoding_name = font_info.get("encoding")
    if encoding_name == 'WinAnsiEncoding':
        encoding_map = pdfminer.encodingdb.get_encoding('winansi')
        for ccode in range(256):
            unicode_char = encoding_map.get(ccode)
            if unicode_char:
                char_map[ccode] = ord(unicode_char)
        return char_map

    # If all else fails, return an empty mapping
    return char_map

# Removed redundant code outside the function

def to_markdown(pdf_file):
    doc = fitz.open(pdf_file)
    md_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_dict = page.get_text("rawdict")
        page_text = ""
        font_char_maps = {}  # Cache for font character maps
        for block in raw_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_name = span['font']
                        font_size = span['size']
                        # Get the font's xref
                        font_xref = None
                        for font in page.get_fonts():
                            if font[3] == font_name:
                                font_xref = font[0]
                                break
                        if font_xref is None:
                            continue  # Skip if font not found
                        # Get or create the character map for this font
                        if font_xref in font_char_maps:
                            char_map = font_char_maps[font_xref]
                        else:
                            char_map = get_char_map(doc, font_xref)
                            font_char_maps[font_xref] = char_map
                        # Process each character
                        for char in span['chars']:
                            ccode = char['ccode']
                            unicode_value = char_map.get(ccode)
                            if unicode_value:
                                corrected_char = chr(unicode_value)
                            else:
                                corrected_char = char['c']  # Fallback to extracted character
                            page_text += corrected_char
                    page_text += '\n'  # Line break
        md_text += page_text + '\n'  # Page break
    return md_text
                        # font_size = span['size']  # Removed unused variable
