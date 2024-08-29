from is_regex import IsRemovedIf, IsNewLineIf, IsMetaDataIf, IsNoise
from meta_logger import MetaLogger
import re

from pathlib import Path
rd = Path(__file__).resolve().parents[2]
save_dir = rd / 'saved'
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
    
class AppliedRegex:
    def __init__(self, text):
        """
        Initialize the AppliedRegex class.
        :param text: str - The text to be manipulated with regex patterns.
        """
        self.text = text

    def _apply_pattern(self, method, *args, **kwargs) -> str:
        """
        Apply a regex pattern and handle logging if needed.
        :param method: callable - The method to apply.
        :return: str - The processed text.
        """
        regex_checker = IsRemovedIf(self.text)
        if method(regex_checker, *args, **kwargs):
            MetaLogger.logformative(self.text, str(save_dir))
            return ''
        return self.text
        
    def _apply_text_format(self, method, *args, **kwargs) -> str:
        regex_checker = IsNewLineIf(self.text)
        if method(regex_checker, *args, **kwargs):
            MetaLogger.logformative(self.text, str(save_dir))
            return ''.join('\n' + self.text + '\n')

    def applied_title(self) -> str:
        """
        Apply the title regex pattern.
        :return: str - The text with the title regex pattern applied.
        """
        return self._apply_pattern(lambda rc: rc.is_title(), "title")

    def applied_arabic(self) -> str:
        """
        Apply the Arabic regex pattern.
        :return: str - The text with the Arabic regex pattern applied.
        """
        return self._apply_pattern(lambda rc: rc.is_arabic_or_diacritics(), "arabic")

    def applied_pua(self) -> str:
        """
        Apply the PUA regex pattern.
        :return: str - The text with the PUA regex pattern applied.
        """
        return self._apply_pattern(lambda rc: rc.is_pua(), "pua")

    def applied_unwanted(self) -> str:
        """
        Apply the unwanted regex pattern.
        :return: str - The text with the unwanted regex pattern applied.
        """
        return self._apply_pattern(lambda rc: rc.is_unwanted(), "unwanted")

    def applied_bullet(self) -> str:
        """
        Apply the bullet regex pattern.
        :return: str - The text with the bullet regex pattern applied.
        """
        return self._apply_pattern(lambda rc: rc.is_bullet(), "bullet")

    def applied_dots(self) -> str:
        """
        Apply the dots regex pattern.
        :return: str - The text with the dots regex pattern applied.
        """
        return self._apply_pattern(lambda rc: rc.is_dot(), "dots")

    def applied_citation(self) -> str:
        """
        Apply the citation regex pattern and log the citation number.
        :return: str - The text with citation regex pattern applied.
        """
        regex_checker = IsNewLineIf(self.text)
        if regex_checker.is_citation():
            MetaLogger.logformative(self.text, "citation_file_name")
            return f'- - - -{self.text},- - - -'
        return self.text

    def applied_hyphenated(self) -> str:
        """
        Handle hyphenated words at line breaks and reformat the text.
        :return: str - The text with hyphenated words reformatted.
        """
        lines = self.text.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            if line.endswith('-') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[a-zA-Z]', next_line):
                    line = line[:-1] + next_line
                    i += 1
            new_lines.append(line)
            i += 1
        return '\n'.join(new_lines).strip()

    def applied_analyze(self) -> str:
        """
        Analyze the text for noise and log the analysis results.
        :return: str - An empty string if analysis is performed.
        """
        regex_checker = IsNoise(self.text)
        analyze_results = {
            "is_non_alphanumeric": regex_checker.is_non_alphanumeric(),
            "is_stand_alone_digit": regex_checker.is_stand_alone_digit(),
            "is_non_ascii": regex_checker.is_non_ascii(),
            "is_excessive_punctuation": regex_checker.is_excessive_punctuation(),
            "is_repetitive_character": regex_checker.is_repetitive_character()
        }
        
        if any(analyze_results.values()):
            MetaLogger.logformative(f"Analysis Results:\n{analyze_results}", "analyze_results_file_name")

            stats = {
                "total_checks": len(analyze_results),
                "detected_counts": {key: value for key, value in analyze_results.items() if value},
                "total_detected": sum(analyze_results.values()),
                "detection_rate": (sum(analyze_results.values()) / len(analyze_results) * 100) if len(analyze_results) > 0 else 0
            }
            MetaLogger.logformative(f"Statistical Data:\n{stats}", "stats_file_name")
            return ''
        
        return self.text

class AggregatedRegex:
    def __init__(self, text, file_path=save_dir):
        """
        Initialize the AggregatedRegex class.
        :param text: str - The text to be manipulated with regex patterns.
        :param file_path: str - Optional path to a file containing text to be processed.
        """
        self.text = text
        self.file_path = file_path

    def process_text(self) -> str:
        """
        Process the text based on regex checks and controls.
        :return: str - The processed text based on regex checks.
        """
        regex_checker = IsNewLineIf(self.text)
        process_results = {
            "is_arabic": regex_checker.is_arabic_or_diacritics(),
            "is_pua": regex_checker.is_pua(),
            "is_unwanted": regex_checker.is_unwanted(),
            "is_bullet": regex_checker.is_bullet(),
            "is_excessive_punctuation": regex_checker.is_excessive_punctuation(),
            "is_repetitive_character": regex_checker.is_repetitive_character(),
            "is_stand_alone_digit": regex_checker.is_stand_alone_digit(),
            "is_title": regex_checker.is_title(),
            "is_citation": regex_checker.is_citation(),
            "is_hyphenated": regex_checker.is_hyphenated(),
            "is_dot": regex_checker.is_dot(),
            "is_non_ascii": regex_checker.is_non_ascii(),
            "is_non_alphanumeric": regex_checker.is_non_alphanumeric()
        }
        if process_results["is_arabic"] or process_results["is_pua"]:
            return AppliedRegex(self.text).applied_analyze()

        return self.text