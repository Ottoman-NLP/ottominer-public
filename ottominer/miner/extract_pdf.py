import fitz  # PyMuPDF
from typing import Tuple, Optional, List
from normalization import Normalization  # Import the updated Normalization class

class MostCommonFontExtraction:
    def __init__(self, pdf_document: fitz.Document):
        """
        Initialize the MostCommonFontExtraction class.

        :param pdf_document: The PDF document to analyze for font sizes.
        """
        self.pdf_document = pdf_document
        self.main_font_size, self.next_size = self._analyze_fonts()
        self.normalizer = Normalization()  # Instantiate Normalization

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
            arabic_buffer = []
            processing_arabic = False

            for page in pdf:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block.keys():
                        for line in block['lines']:
                            line_text = self._process_line(line['spans'], arabic_buffer, processing_arabic)
                            extracted_text += line_text + "\n"

            if arabic_buffer:
                extracted_text += "\n" + " ".join(arabic_buffer[::-1]) + "\n"

        return extracted_text

    def _process_line(self, spans: List[dict], arabic_buffer: List[str], processing_arabic: bool) -> str:
        """
        Process a line of text spans and extract relevant text.

        :param spans: List of spans to process.
        :param arabic_buffer: Buffer to store Arabic text.
        :param processing_arabic: Flag indicating if Arabic text processing is active.
        :return: Processed line text.
        """
        line_text = []
        for span in spans:
            text = span['text'].strip()
            font_size = span['size']
            is_bold = 'Bold' in span['font'] or 'Black' in span['font']
            contains_alpha = any(c.isalpha() for c in text)
            is_numeric_only_bold = is_bold and text.isnumeric()

            if self.is_turkish(text):
                processing_arabic = True
                text = text.strip("()")
                normalized_text = self.normalize_chars(text)
                arabic_buffer.append(normalized_text)
            elif processing_arabic:
                if arabic_buffer:
                    arabic_text = " ".join(arabic_buffer)
                    line_text.append("\n" + arabic_text[::-1] + "\n")
                    arabic_buffer.clear()
                processing_arabic = False
                line_text.append(text)
            elif (font_size in [self.main_font_size, self.next_size] and contains_alpha and not is_numeric_only_bold):
                line_text.append(text)

        return " ".join(line_text)

    def is_turkish(self, text: str) -> bool:
        """
        Check if the text is Turkish.
        (Placeholder function - implement your logic)
        """
        # Implement your Turkish text detection logic
        return False

    def normalize_chars(self, text: str) -> str:
        """
        Normalize characters in the text using the Normalization class.
        
        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        text = self.normalizer.normalize_char(text)
        text = self.normalizer.normalize_unknown_char(text)
        return text