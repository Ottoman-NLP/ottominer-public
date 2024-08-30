from is_regex import IsRemovedIf, IsNewLineIf
from pathlib import Path
from meta_logger import MetaLogger


rd = Path(__file__).resolve().parents[2]
save_dir = rd / 'saved'
save_dir.mkdir(parents=True, exist_ok=True)

class AppliedRegex:
    def __init__(self, text: str):
        """
        Initialize the AppliedRegex class.
        
        :param text: The text to be manipulated with regex patterns.
        """
        self.text = text

    def apply_format_or_remove(self) -> str:
        """
        Decide whether to format or remove text based on regex patterns.
        
        :return: The formatted or removed text based on checks.
        """
        remove_checker = IsRemovedIf(self.text)
        format_checker = IsNewLineIf(self.text)
        
        if self._apply_remove(remove_checker):
            MetaLogger.logformative(self.text, str(save_dir))
            return ''  # Text is removed based on the condition
        
        if self._apply_format(format_checker):
            MetaLogger.logformative(self.text, str(save_dir))
            return '\n' + self.text + '\n'  # Newline formatted text
        
        return self.text  # No changes if neither condition is met

    def _apply_remove(self, checker: IsRemovedIf) -> bool:
        """
        Apply the remove pattern checks.
        
        :param checker: Instance of IsRemovedIf.
        :return: True if any remove condition matches, else False.
        """
        return any([
            checker.is_arabic_or_diacritics(),
            checker.is_pua(),
            checker.is_unwanted(),
            checker.is_bullet(),
            checker.is_excessive_punctuation(),
            checker.is_repetitive_character(),
            checker.is_stand_alone_digit()
        ])

    def _apply_format(self, checker: IsNewLineIf) -> bool:
        """
        Apply the format pattern checks.
        
        :param checker: Instance of IsNewLineIf.
        :return: True if any format condition matches, else False.
        """
        return any([
            checker.is_title(),
            checker.is_citation(),
            checker.is_hyphenated(),
            checker.is_dot()
        ])