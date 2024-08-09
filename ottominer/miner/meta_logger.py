from datetime import datetime
from pathlib import Path
import sys

rd = Path(__file__).resolve().parents[2]
applied_function_pack = rd / "mining" / "miner"
sys.path.append(str(applied_function_pack))

from applied_function import AggregatedRegex

class MetaLogger:
    """
    Logs various messages to a file and tracks noise data from text analysis.
    :parameters:
    log_file: str - The name of the log file to write to. Default is 'meta_log.txt'.
    log_path: str - The path to the log file. Default is 'logs/meta_log.txt'.
    :methods:
    is_informative: Logs general or error messages.
    log_noise_data: Logs noise data from text analysis.
    """

    def __init__(self, log_file='meta_log.txt'):
        self.log_file = log_file
        self.log_path = Path(__file__).resolve().parents[3] / "logs" / log_file
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensure the log directory exists."""
        if not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True)

    def is_informative(self, informative: str, error=False):
        """
        Logs the message to the log file.
        :parameters:
        informative: str - The informative message to be logged.
        error: bool - Whether the message is an error. Default is False.
        :returns:
        None
        """
        log_message = f"ERROR: {informative}" if error else f"INFO: {informative}"
        with open(self.log_path, 'a') as f:
            f.write(f"{log_message}\n")
        print(log_message)

    def logformative(self, aggregated_regex: AggregatedRegex, file_name: str):
        """
        Logs noise-related data obtained from the AggregatedRegex instance.
        :parameters:
        aggregated_regex: AggregatedRegex - An instance of AggregatedRegex class with analyzed text.
        file_name: str - The name of the file being analyzed.
        :returns:
        None
        """
        analyze_results = aggregated_regex.analyze_text()
        log_message = f"--- Noise Data Analysis for File: {file_name} ---\n"
        log_message += f"Analysis Timestamp: {datetime.now()}\n"
        log_message += f"File Path: {aggregated_regex.file_path}\n"

        for key, value in analyze_results.items():
            log_message += f"{key}: {value}\n"

        log_message += "\n"
        
        # Write log to file and print to console
        with open(self.log_path, 'a') as f:
            f.write(log_message)
        print(log_message)
