import re

class IsRegex:

    def __init__(self, text):
        """
        :param text: str - The text to be manipulated given the regex patterns
        """
        self.text = text

    def is_arabic_or_diacritics(self):
        return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', self.text))
    
    def is_pua(self):
        return bool(re.search(r'[\uE000-\uF8FF]', self.text))
    
    def is_unwanted(self):
        return bool(re.search(r'[()،؛,]', self.text))

    def is_bullet(self):
        return bool(re.search(r'^\s*\d+\.\s+', self.text))
    
    def is_dots(self):
        return bool(re.search(r'^\s*\.*\s*$', self.text))
    
    def is_citation(self):
        return bool(re.search(r'^\s*\[\s*\d*\s*\]\s*$', self.text))
    
    def is_hyphenated(self):
        return bool(re.search(r'^\s*[a-zA-Z]+-\s*$', self.text))
    
    def is_title(self):
        return bool(re.search(r'^[A-Z\s\'.-]+$', self.text))
    
class IsNoise:

    def __init__(self, text):
        self.text = text

    def is_non_alphanumeric(self):
        return bool(re.search(r'[^\w\s,.?!]', self.text))
    
    def is_stand_alone_digit(self):
        return bool(re.search(r'^\d+$', self.text))
    
    def is_non_ascii(self):
        return bool(re.search(r'[^\x00-\x7F]', self.text))
    
    def is_excessive_punctuation(self):
        return bool(re.search(r'[!.,?;:]{2,}', self.text))
    
    def is_repetitive_character(self):
        return bool(re.search(r'\s{2,}', self.text))
    
class IsRecursive:

    def __init__(self, text):
        self.text = text

    def is_rmcas(self):
        return re.findall(r'(\bSEBILÜRREŞAD\s*)?\b(CİLD|ADED|SAYFA)\b\s*\d+(-\d+)?', self.text)
