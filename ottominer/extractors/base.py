from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from ..utils.resources import check_system_resources
from ..utils.logger import setup_logger
import logging
from dataclasses import dataclass
import traceback

logger = setup_logger(__name__)

@dataclass
class ExtractionError:
    """Extraction error details."""
    message: str
    file_path: Path
    error_type: str
    details: Optional[dict] = None

class ExtractorException(Exception):
    """Base exception for extraction errors."""
    def __init__(self, error: ExtractionError):
        self.error = error
        super().__init__(str(error))

class BaseExtractor(ABC):

    """base class for all extractors"""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> Dict:
        """Extract text from a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text content
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    def batch_extract(self, file_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Extract text from multiple documents.
        
        Args:
            file_paths: List of paths to documents
            
        Returns:
            Dictionary mapping file paths to extracted content
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass
    
    def validate_file(self, file_path: Union[str, Path]) -> Path:

        """Validate file path and extension.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is not a file
        """
        path = Path(file_path)
        if not path.exists():
            err_msg = f"File does not exist: {file_path}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        
        if not path.is_file():
            err_msg = f"Path is not a file: {file_path}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        
        return path

    def handle_extraction_error(self, e: Exception, file_path: Path) -> None:
        """Handle extraction errors with proper logging."""
        error = ExtractionError(
            message=str(e),
            file_path=file_path,
            error_type=type(e).__name__,
            details={'traceback': traceback.format_exc()}
        )
        self.logger.error(f"Extraction failed: {error}")
        raise ExtractorException(error)
