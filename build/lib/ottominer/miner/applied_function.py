
"""
This module contains the regex patterns - in action - used in the mining process.
"""
from pathlib import Path
from is_regex import IsRegex, IsRecursive, IsNoise
from meta_logger import MetaLogger
import re


class AppliedRegex:

    def __init__(self, text):
        """
        Initializing the AppliedRegex class
        :param text: str - The text to be manipulated given the regex patterns
        """
        self.text = text

    def applied_title(self) -> str: # REFORMAT
        """
        Apply the title regex pattern to the text.
        :return: str - The text with the title regex pattern applied
        """
        regex_checker = IsRegex(self.text)
        meta_logger = MetaLogger()
        if regex_checker.is_title():
            meta_logger.is_informative(self.text)
            return f'----{self.text}----'
        return self.text

    def applied_arabic(self) -> str: # REMOVE
        """
        Apply Arabic regex pattern to the text.
        :return: str - The text with Arabic regex pattern applied -> Removed
        """
        regex_checker = IsRegex(self.text)
        if regex_checker.is_arabic_or_diacritics():
            return ''
        return self.text
    
    def applied_pua(self) -> str: # REMOVE
        """
        Apply PUA regex pattern to the text to remove PUA characters.
        :return: str - The text with PUA regex pattern applied -> Removed
        """
        regex_checker = IsRegex(self.text)
        if regex_checker.is_pua():
            return ''
        return self.text
    
    def applied_unwanted(self) -> str: # REMOVE
        """
        Apply unwanted regex pattern to remove all unwanted instances of punctuation.
        :return: str - The text with unwanted regex pattern applied -> Removed
        """
        regex_checker = IsRegex(self.text)
        if regex_checker.is_unwanted():
            return ''
        return self.text
    
    def applied_bullet(self) -> str: # REMOVE
        """
        Apply bullet regex pattern to remove all bullets.
        :return: str - The text with bullet regex pattern applied -> Removed
        """
        regex_checker = IsRegex(self.text)
        if regex_checker.is_bullet():
            return ''
        return self.text
    
    def applied_dots(self) -> str: # REMOVE
        """
        Apply dots regex pattern to remove all dots.
        :return: str - The text with dots regex pattern applied -> Removed
        """
        regex_checker = IsRegex(self.text)
        if regex_checker.is_dots():
            return ''
        return self.text
    
    def applied_citation(self) -> str: # LOG & REFORMAT
        """
        Apply citation regex pattern to save the citation number in the log file and reformat the text.
        :return: str - The text with citation regex pattern applied -> Saved in log file
        """
        regex_checker = IsRegex(self.text)
        if regex_checker.is_citation():
            MetaLogger.logformative(self.text, "citation_file_name")  # Replace with actual file name
            return f'- - - -{self.text},- - - -'
        return self.text
    
    def applied_hyphenated(self) -> str:
        """
        Handle hyphenated words at line breaks and reformat the text.
        :return: str - The text with hyphenated words reformatted
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
                    lines[i + 1] = ''
            new_lines.append(line)
            i += 1
        
        return '\n'.join(new_lines).strip()

    def applied_analyze(self) -> str:
        """
        Apply analyze regex from analyze_text method to save the text in the log file.
        :return: str - An empty string if analysis is performed
        """
        regex_checker = IsRegex(self.text)
        analyze_results = regex_checker.analyze_text()

        # Check if any noise category is true
        if any(analyze_results.values()):
            # Create a log entry with the statistical data
            MetaLogger.logformative(f"Analysis Results:\n{analyze_results}", "analyze_results_file_name")  # Replace with actual file name

            # Calculate occurrence statistics
            total_checks = len(analyze_results)
            detected_counts = {key: value for key, value in analyze_results.items() if value}
            total_detected = len(detected_counts)

            # Log statistical information
            stats = {
                "total_checks": total_checks,
                "detected_counts": detected_counts,
                "total_detected": total_detected,
                "detection_rate": (total_detected / total_checks) * 100 if total_checks > 0 else 0
            }
            MetaLogger.logformative(f"Statistical Data:\n{stats}", "stats_file_name")  # Replace with actual file name

            return ''
        
        return self.text



class AggregatedRegex:
    
    def __init__(self, text, file_path=None):
        """
        Initializing the AggregatedRegex class
        :param text: str - The text to be manipulated given the regex patterns
        :param file_path: str - Path to a file containing text to be processed
        """
        self.text = text
        self.file_path = file_path

    def process_text(self) -> str:
        """
        Process the text based on regex checks and controls.
        :return: str - The processed text if conditions are met, otherwise the original text
        """
        regex_checker = IsRegex(self.text)

        process_results = {
            "is_arabic": regex_checker.is_arabic_or_diacritics(),
            "is_pua": regex_checker.is_pua(),
            "is_unwanted": regex_checker.is_unwanted(),
            "is_bullet": regex_checker.is_bullet(),
            "is_dots": regex_checker.is_dots(),
            "is_citation": regex_checker.is_citation(),
            "is_hyphenated": regex_checker.is_hyphenated(),
            "is_title": regex_checker.is_title()
        }
        
        if process_results["is_title"]:
            return AppliedRegex(self.text).applied_title()
        elif process_results["is_arabic"]:
            return AppliedRegex(self.text).applied_arabic()
        else:
            return self.text

    def recursive_text(self) -> str:
        """
        Recursively remove undesirable textual instances.
        :return: str - The cleaned text
        """
        recursive_checker = IsRecursive(self.text)
        is_rmcas = recursive_checker.is_rmcas()
        return ' '.join(is_rmcas)
    
    def analyze_text(self) -> dict:
        """
        Analyze the text using defined regex patterns for noise detection.
        :return: dict - The analyzed results
        """
        noise_checker = IsNoise(self.text)
        analyze_results = {
            "is_non_alphanumeric": noise_checker.is_non_alphanumeric(),
            "is_stand_alone_digit": noise_checker.is_stand_alone_digit(),
            "is_non_ascii": noise_checker.is_non_ascii(),
            "is_excessive_punctuation": noise_checker.is_excessive_punctuation(),
            "is_repetitive_character": noise_checker.is_repetitive_character()
        }
        return analyze_results