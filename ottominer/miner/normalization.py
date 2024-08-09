"""
This module contains normalization characters for Ottoman Turkish text processing 
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
    :param text: The text to be normalized.
    :type text: str
    :return: The normalized text.
    :rtype: str
    """

    def __init__(self):
        self.char_setup = char_setup
        self.unknown_char_setup = unknown_char_setup

    def normalize_char(self, text):
        """
        only normalizes characters
        """
        for char, replacement in self.char_setup.items():
            text = ''.join(char_setup.get(char, char) for char in text)
            return text
    
    def normalize_unknown_char(self, text):
        """
        normalizes and returns analytical information about the text on unknown characters
        """
        for char, replacement in self.unknown_char_setup.items():
            text = ''.join(unknown_char_setup.get(char, char) for char in text)
            return text


"""    def normalize_characters(self, text):
        for char, replacement in self.char_setup.items():
            text = text.replace(char, replacement)
        return text"""