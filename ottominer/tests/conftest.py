import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock
from reportlab.pdfgen import canvas

# Use absolute imports
from ottominer.utils.progress import ProgressTracker
from ottominer.core.config import Config

@pytest.fixture(scope="session")
def test_env():
    """Create test environment directories."""
    temp_dir = Path(tempfile.mkdtemp())
    dirs = {
        'config': temp_dir / 'config',
        'output': temp_dir / 'output',
        'cache': temp_dir / 'cache',
        'logs': temp_dir / 'logs',
        'fdata': temp_dir / 'fdata'
    }
    
    # Create all directories
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    yield dirs
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a valid sample PDF for testing."""
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test PDF Document")
    c.drawString(100, 700, "This is a test PDF file.")
    c.save()
    return pdf_path

@pytest.fixture
def mock_progress():
    """Create mock progress for testing."""
    mock = MagicMock()
    mock.task.return_value = 0
    mock.update.return_value = None
    return mock

@pytest.fixture
def temp_pdfs(tmp_path):
    """Create multiple valid test PDFs."""
    paths = []
    for i in range(3):
        pdf_path = tmp_path / f"test_{i}.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, f"Test Document {i}")
        c.drawString(100, 700, "Sample content for testing")
        c.save()
        paths.append(pdf_path)
    return paths

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'pdf_extraction': {
            'extract_images': False,
            'dpi': 300,
            'margins': (50, 50, 0, 0),
            'table_strategy': 'lines_strict',
            'fontsize_limit': 4,
            'workers': 2,
            'batch_size': 10
        }
    }

@pytest.fixture
def small_test_pdf(tmp_path):
    """Create a small test PDF for faster testing."""
    pdf_path = tmp_path / "small_test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test PDF Document")
    c.drawString(100, 700, "This is a small test file.")
    c.drawString(100, 650, "It contains minimal content.")
    c.save()
    return pdf_path