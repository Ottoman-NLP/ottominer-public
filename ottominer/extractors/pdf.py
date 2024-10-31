from pathlib import Path
from typing import Dict, Union, List
import pymupdf4llm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
from functools import partial
from rich.progress import Progress, SpinnerColumn
import logging
import os

# Fix imports to use absolute paths
from ..core.config import Config
from ..utils.logger import setup_logger
from ..utils.progress import ProgressTracker
from ..utils.decorators import handle_exceptions
from ..utils.resources import check_system_resources
from .base import BaseExtractor  # This is correct as it's in same directory

logger = setup_logger(__name__)

class PDFExtractor(BaseExtractor):
    """PDF extraction implementation with markdown conversion."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.config = config.get('pdf_extraction', {}) if config else {}
        # Filter only supported pymupdf4llm parameters
        self.pdf_config = {
            'dpi': self.config.get('dpi', 300),
            'margins': self.config.get('margins', (50, 50, 0, 0)),
            'table_strategy': self.config.get('table_strategy', 'lines_strict'),
            'fontsize_limit': self.config.get('fontsize_limit', 4)
        }
        self.progress = ProgressTracker()
    
    def batch_extract(self, file_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Implement required abstract method."""
        return {str(path): self.extract(path) for path in file_paths}
    
    @handle_exceptions
    def extract(self, file_path: Union[str, Path], timeout: int = 30) -> str:
        """Extract text from PDF with progress tracking."""
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            transient=True
        ) as progress:
            task = progress.add_task(f"Processing {file_path}...", total=1)
            try:
                result = self._extract_impl(file_path)
                progress.update(task, advance=1)
                return result
            except Exception as e:
                progress.update(task, description=f"Error: {str(e)}")
                raise
    
    def _extract_impl(self, file_path):
        """Actual extraction implementation."""
        path = self.validate_file(file_path)
        return self._convert_to_markdown(path)
    
    def _convert_to_markdown(self, file_path: Path) -> str:
        """Convert PDF to markdown with progress tracking."""
        try:
            if not self._is_valid_pdf(file_path):
                raise ValueError(f"Invalid or corrupted PDF file: {file_path}")
                
            md_text = pymupdf4llm.to_markdown(
                str(file_path),
                **self.pdf_config  # Use filtered config
            )
            return md_text
            
        except Exception as e:
            logger.error(f"Error converting PDF to markdown: {e}")
            raise
    
    def _is_valid_pdf(self, file_path: Path) -> bool:
        """Check if file is a valid PDF."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                return header.startswith(b'%PDF-')
        except Exception:
            return False
    
    def _save_output(self, content: str, source_path: Path) -> Path:
        """Save extracted content to output directory."""
        output_dir = Path(self.config.get('output_dir', 'output'))
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"{source_path.stem}.md"
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Saved output to: {output_path}")
        return output_path
    
    def save_output(self, content: str, output_path: Union[str, Path]) -> None:
        """Save extracted content to file.
        
        Args:
            content: Content to save
            output_path: Path to save to
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')


class ParallelPDFExtractor(PDFExtractor):
    """Parallel PDF extraction implementation."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.max_workers = self.config.get('workers', max(2, os.cpu_count() // 2))

    def _extract_single(self, file_path: Path, progress=None, task_id=None) -> str:
        """Extract text from a single PDF with optional progress tracking."""
        try:
            result = self._convert_to_markdown(file_path)
            if progress and task_id:
                progress.update(task_id, advance=1)
            return result
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None

    @handle_exceptions
    def batch_extract(self, file_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Extract text from multiple PDFs in parallel."""
        file_paths = [self.validate_file(f) for f in file_paths]
        results = {}

        with self.progress as progress:
            task_id = progress.add_task(
                f"Processing {len(file_paths)} PDFs", 
                total=len(file_paths)
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files for processing
                future_to_path = {
                    executor.submit(
                        self._extract_single, 
                        path,
                        progress,
                        task_id
                    ): path 
                    for path in file_paths
                }
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[str(path)] = result
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        results[str(path)] = None

        return results or None
    