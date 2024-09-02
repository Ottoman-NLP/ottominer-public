import fitz
from typing import Tuple, Optional
from normalization import Normalization

class MostCommonFontExtraction:
    def __init__(self, pdf_document: fitz.Document):
        """
        Initialize the MostCommonFontExtraction class.

        :param pdf_document: The PDF document to analyze for font sizes.
        """
        self.pdf_document = pdf_document
        self.main_font_size, self.next_size = self._analyze_fonts()
        self.normalizer = Normalization()

    def _analyze_fonts(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Analyze the fonts used in the PDF document and identify the most common font sizes.

        :return: A tuple containing the most common font size and the second most common font size.
        """
        font_data = self._extract_font_sizes()
        if font_data:
            sorted_fonts = sorted(font_data.items(), key=lambda item: item[1], reverse=True)
            main_font_size = sorted_fonts[0][0]
            next_size = sorted_fonts[1][0] if len(sorted_fonts) > 1 else main_font_size
            return main_font_size, next_size
        return None, None

    def _extract_font_sizes(self) -> dict:
        """
        Extract font sizes from each page of the PDF document.

        :return: A dictionary containing font sizes and their frequencies.
        """
        font_data = {}
        for page in self.pdf_document:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block.keys():
                    for line in block['lines']:
                        for span in line['spans']:
                            font_size = span['size']
                            font_data[font_size] = font_data.get(font_size, 0) + 1
        return font_data

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using the most common font sizes.

        :param pdf_path: Path to the PDF file.
        :return: Extracted text as a string.
        """
        with fitz.open(pdf_path) as pdf:
            extracted_text = ""
            for page in pdf:
                blocks = page.get_text("dict")["blocks"]
            print(blocks)


        return extracted_text
    
if __name__ == "__main__":
    ext_text = MostCommonFontExtraction.extract_text("../saved/pdfs/evliya_celebi.pdf")
    print(ext_text)