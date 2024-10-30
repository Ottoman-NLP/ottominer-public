import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_environment():
    """Create test environment with temporary directories"""
    test_dir = Path(tempfile.mkdtemp())
    
    # Create necessary subdirectories
    dirs = {
        'input': test_dir / 'input',
        'output': test_dir / 'output',
        'cache': test_dir / 'cache'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield dirs
    
    # Cleanup
    shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def sample_pdf(test_environment):
    """Create a sample PDF file"""
    pdf_path = test_environment['input'] / 'test.pdf'
    pdf_path.touch()
    return pdf_path 