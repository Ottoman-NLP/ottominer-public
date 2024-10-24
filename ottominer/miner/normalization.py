"""
This module contains normalization functions for Ottoman Turkish text processing 
for letters and unknown characters.
"""

char_setup = {

    'â': 'a', 'Â': 'A',
    'ā': 'a', 'Ā': 'A',
    "ê": "e",
    "ē": "e", "Ē": "E",
    "å": "a", "Å": "A",
    'î': 'i', 'Î': 'I',
    "ī": "i", "Ī": "I", 
    'û': 'u', 'Û': 'U',
    "ô": "ö",
    "ō": "o",  "Ō": "Ö",
    
}

unknown_char_setup = {
     "": "",
     "": "",
     "œ": "i",
     "†": "u",
     "¢": "i",
     "“": '"', "”": '"', "’": "'", "–": "-", "—": "-", "…": "...", "": "",
}

class Normalization:
    """
    Class for normalizing characters in Ottoman Turkish text.

    :return: The normalized text.
    :rtype: str
    """
    def __init__(self):
        self.char_setup = char_setup
        self.unknown_char_setup = unknown_char_setup

    def normalize_char(self, text: str) -> str:
        """
        Normalizes regular characters in the text.

        :param text: The text to be normalized.
        :type text: str
        :return: The text with normalized characters.
        :rtype: str
        """
        for char, replacement in self.char_setup.items():
            text = text.replace(char, replacement)
        return text
    
    def normalize_unknown_char(self, text: str) -> str:
        """
        Normalizes unknown characters in the text and returns the processed text.

        :param text: The text to be normalized.
        :type text: str
        :return: The text with unknown characters normalized.
        :rtype: str
        """
        for char, replacement in self.unknown_char_setup.items():
            text = text.replace(char, replacement)
        return text