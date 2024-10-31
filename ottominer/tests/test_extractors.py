import pytest
from pathlib import Path
from ottominer.extractors.pdf import PDFExtractor, ParallelPDFExtractor
from reportlab.pdfgen import canvas
import time
from ottominer.utils.progress import ProgressTracker

def create_test_pdf(tmp_path, content="Test Document", filename="test.pdf"):
    """Create a small test PDF file."""
    pdf_path = tmp_path / filename
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, content)
    c.save()
    return pdf_path

@pytest.fixture
def test_pdfs(tmp_path):
    """Create multiple test PDFs."""
    return [
        create_test_pdf(tmp_path, f"Test Doc {i}", f"test_{i}.pdf")
        for i in range(3)
    ]

@pytest.fixture
def pdf_config():
    """Test configuration for PDF extraction."""
    return {
        'pdf_extraction': {
            'dpi': 300,
            'margins': (50, 50, 0, 0),
            'table_strategy': 'lines_strict',
            'fontsize_limit': 4,
            'workers': 2,
            'batch_size': 10
        }
    }

class TestBaseExtractor:
    @pytest.mark.timeout(5)
    def test_validate_file_exists(self, tmp_path):
        pdf_path = create_test_pdf(tmp_path)
        extractor = PDFExtractor()
        assert extractor.validate_file(pdf_path) == pdf_path

    @pytest.mark.timeout(5)
    def test_validate_file_not_exists(self, tmp_path):
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.validate_file(tmp_path / "nonexistent.pdf")

    @pytest.mark.timeout(5)
    def test_validate_file_not_file(self, tmp_path):
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.validate_file(tmp_path)

class TestPDFExtractor:
    @pytest.mark.timeout(5)
    def test_extract_single_pdf(self, tmp_path, pdf_config):
        pdf_path = create_test_pdf(tmp_path)
        extractor = PDFExtractor(pdf_config)
        result = extractor.extract(pdf_path)
        assert isinstance(result, str)
        assert "Test Document" in result

    @pytest.mark.timeout(5)
    def test_save_output(self, tmp_path):
        extractor = PDFExtractor()
        output = "Test content"
        output_file = tmp_path / "output.txt"
        extractor.save_output(output, output_file)
        assert output_file.read_text(encoding='utf-8') == output

    @pytest.mark.timeout(5)
    def test_extract_invalid_pdf(self, tmp_path):
        pdf_path = tmp_path / "invalid.pdf"
        pdf_path.write_bytes(b"Not a PDF")
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.extract(pdf_path)

class TestParallelPDFExtractor:
    @pytest.mark.timeout(10)
    def test_batch_extract(self, test_pdfs):
        """Test batch extraction of PDFs."""
        extractor = ParallelPDFExtractor()
        results = extractor.batch_extract(test_pdfs)
        
        # Verify results
        assert results is not None
        assert len(results) == len(test_pdfs)
        
        # Check content of successful extractions
        successful_results = [v for v in results.values() if v is not None]
        assert len(successful_results) > 0
        assert all("Test Doc" in v for v in successful_results)

    @pytest.mark.timeout(5)
    def test_resource_management(self):
        extractor = ParallelPDFExtractor()
        assert extractor.max_workers > 0

@pytest.mark.timeout(10)
def test_integration(test_pdfs):
    """Integration test for PDF extraction."""
    config = {
        'pdf_extraction': {
            'dpi': 300,
            'margins': (50, 50, 0, 0),
            'table_strategy': 'lines_strict',
            'fontsize_limit': 4,
            'workers': 2
        }
    }

    # Force stop any existing progress
    ProgressTracker().force_stop()

    # Test single extraction
    extractor = PDFExtractor(config)
    result = extractor.extract(test_pdfs[0])
    assert isinstance(result, str)
    assert "Test Doc 0" in result

    # Force stop before parallel extraction
    ProgressTracker().force_stop()
    time.sleep(0.1)  # Give time for cleanup

    # Test parallel extraction
    parallel = ParallelPDFExtractor(config)
    results = parallel.batch_extract(test_pdfs)
    assert results is not None
    assert len(results) == len(test_pdfs)
    
    # Check content of successful extractions
    successful_results = [v for v in results.values() if v is not None]
    assert len(successful_results) > 0
    assert all("Test Doc" in v for v in successful_results)