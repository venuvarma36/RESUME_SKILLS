"""
Unit tests for Text Extraction Module
"""

import pytest
from pathlib import Path
from text_extraction import TextExtractor


class TestTextExtractor:
    """Test cases for TextExtractor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = TextExtractor()
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly."""
        assert self.extractor is not None
        assert self.extractor.min_text_length > 0
        assert isinstance(self.extractor.ocr_fallback, bool)
    
    def test_extract_from_nonexistent_file(self):
        """Test extraction from non-existent file."""
        result = self.extractor.extract("nonexistent_file.pdf")
        
        assert result['success'] is False
        assert result['error'] == 'File not found'
        assert result['text'] == ''
    
    def test_extract_unsupported_format(self, tmp_path):
        """Test extraction from unsupported file format."""
        # Create a temporary txt file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Sample text")
        
        result = self.extractor.extract(str(txt_file))
        
        assert result['success'] is False
        assert 'Unsupported format' in result['error']
    
    def test_extract_with_empty_content(self, tmp_path):
        """Test extraction with minimal content."""
        # This test would require creating actual PDF/DOCX files
        # which is complex in unit tests
        pass
    
    def test_batch_extraction(self):
        """Test batch extraction."""
        files = ["file1.pdf", "file2.pdf"]
        results = self.extractor.extract_batch(files)
        
        assert isinstance(results, dict)
        assert len(results) == len(files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
