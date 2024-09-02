import re

class RegexPatterns:
    ARABIC = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]'
    PUA = r'[\uE000-\uF8FF]'
    UNWANTED = r'[()،؛,]'
    BULLET = r'^\s*\d+\.\s+'
    EXCESSIVE_PUNCTUATION = r'[!.,?;:]{2,}'
    REPETITIVE_CHAR = r'\s{2,}'
    STAND_ALONE_DIGIT = r'^\d+$'
    TITLE = r'^[A-Z\s\'.-]+$'
    CITATION = r'^\s*\[\s*\d*\s*\]\s*$'
    HYPHENATED = r'^\s*[a-zA-Z]+-\s*$'
    DOT = r'[\w][^.]+'
    NON_ASCII = r'[^\x00-\x7F]'
    NON_ALPHANUMERIC = r'[^\w\s,.?!]'

class IsRemovedIf:
    def __init__(self, text: str):
        self.text = text

    def matches(self, pattern: str) -> bool:
        """
        Check if the text matches the given pattern.

        :param pattern: Regex pattern to match.
        :return: True if pattern matches, False otherwise.
        """
        return bool(re.search(pattern, self.text))

    def is_arabic_or_diacritics(self) -> bool:
        return self.matches(RegexPatterns.ARABIC)
    
    def is_pua(self) -> bool:
        return self.matches(RegexPatterns.PUA)
    
    def is_unwanted(self) -> bool:
        return self.matches(RegexPatterns.UNWANTED)

    def is_bullet(self) -> bool:
        return self.matches(RegexPatterns.BULLET)
    
    def is_excessive_punctuation(self) -> bool:
        return self.matches(RegexPatterns.EXCESSIVE_PUNCTUATION)
    
    def is_repetitive_character(self) -> bool:
        return self.matches(RegexPatterns.REPETITIVE_CHAR)
    
    def is_stand_alone_digit(self) -> bool:
        return self.matches(RegexPatterns.STAND_ALONE_DIGIT)
class IsNewLineIf:
    """
    Class for checking patterns that would result in adding a newline.
    """
    def is_title(self) -> bool:
        return self.matches(RegexPatterns.TITLE)
    
    def is_citation(self) -> bool:
        return self.matches(RegexPatterns.CITATION)
    
    def is_hyphenated(self) -> bool:
        return self.matches(RegexPatterns.HYPHENATED)
    
    def is_dot(self) -> bool:
        return self.matches(RegexPatterns.DOT)

class IsMetaDataIf:
    """
    Class for checking metadata related patterns.
    """
    def is_non_ascii(self) -> bool:
        return self.matches(RegexPatterns.NON_ASCII)

    def is_non_alphanumeric(self) -> bool:
        return self.matches(RegexPatterns.NON_ALPHANUMERIC)