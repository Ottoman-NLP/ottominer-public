from is_regex import IsRemovedIf, IsNewLineIf
from pathlib import Path
from meta_logger import MetaLogger
from normalization import Normalization
import re

# Set up the save directory for logging
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
        self.normalizer = Normalization()  # Initialize Normalization class

    def apply_format_or_remove(self) -> str:
        """
        Decide whether to format or remove text based on regex patterns.
        
        :return: The formatted or removed text based on checks.
        """
        # Normalize text
        self.text = self.normalizer.normalize_char(self.text)
        self.text = self.normalizer.normalize_unknown_char(self.text)

        # Apply formatting and removal checks
        remove_checker = IsRemovedIf(self.text)
        format_checker = IsNewLineIf(self.text)

        if self._apply_remove(remove_checker):
            MetaLogger.logformative(self.text, str(save_dir))
            print(f"Text removed due to matching remove conditions.")
            return ''  # Text is removed based on the condition
        
        if self._apply_format(format_checker):
            MetaLogger.logformative(self.text, str(save_dir))
            formatted_text = '\n' + self.text.strip() + '\n'
            formatted_text = self.remove_empty_lines(formatted_text)
            formatted_text = self.handle_special_line_cases(formatted_text)
            print(f"Formatted text: {formatted_text[:30]}...")
            return formatted_text  # Newline formatted text
        
        print(f"No change in text: {self.text[:30]}...")
        return self.text  # No changes if neither condition is met

    def _apply_remove(self, checker: IsRemovedIf) -> bool:
        """
        Apply the remove pattern checks.
        
        :param checker: Instance of IsRemovedIf.
        :return: True if any remove condition matches, else False.
        """
        checks = {
            'is_arabic_or_diacritics': checker.is_arabic_or_diacritics(),
            'is_pua': checker.is_pua(),
            'is_unwanted': checker.is_unwanted(),
            'is_bullet': checker.is_bullet(),
            'is_excessive_punctuation': checker.is_excessive_punctuation(),
            'is_repetitive_character': checker.is_repetitive_character(),
            'is_stand_alone_digit': checker.is_stand_alone_digit(),
        }

        for check, result in checks.items():
            print(f"Check {check}: {result}")

        return any(checks.values())

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

    def remove_empty_lines(self, text: str) -> str:
        """
        Removes empty lines from the text.
        
        :return: Text without empty lines.
        """
        cleaned_text = "\n".join([line for line in text.splitlines() if line.strip()])
        print(f"Text after removing empty lines: {cleaned_text[:30]}...")
        return cleaned_text

    def handle_special_line_cases(self, text: str) -> str:
        """
        Handles specific line conditions, such as merging hyphenated lines and handling titles.
        
        :param text: Text to process.
        :return: Processed text.
        """
        lines = text.splitlines()
        cleaned_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip single character lines that are not alphanumeric
            if len(line) == 1 and not line.isalnum():
                continue
            
            # Skip lines that are stand-alone digits
            if re.match(r'^\d+$', line):
                continue

            # Handle potential titles
            if re.match(r'^[A-ZİĞÜŞÖÇ]+\s*$', line) or re.match(r'^[A-ZİĞÜŞÖÇ\-]+$', line) or re.match(r'^[A-ZİĞÜŞÖÇ]+(?:-[a-z]+)?$', line):
                title_lines = [line]
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line.isupper() or re.match(r'^[A-ZİĞÜŞÖÇ]+(?:-[a-z]+)?$', next_line):
                        title_lines.append(next_line)
                        lines[j] = ''  # Remove the next line since it's joined
                    else:
                        break
                title = ' '.join(title_lines)
                cleaned_lines.append(f"Başlık: {title}")
                continue

            # Handle hyphenated lines
            if line.endswith('-') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[a-zA-Z]', next_line):
                    line = line[:-1] + next_line
                    lines[i + 1] = ''  # Remove the next line since it's joined

            # Join words with spaces before '’'
            line = re.sub(r'\s+’\s*', '’', line)

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def split_into_sentences(self, text: str) -> list:
        """
        Splits text into sentences based on punctuation marks.
        
        :param text: Text to split.
        :return: List of sentences.
        """
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|:)\s')
        sentences = sentence_endings.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]