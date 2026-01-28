"""
Unit tests for Text Preprocessing Module
"""

import pytest
from preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = TextPreprocessor(download_nltk_data=False)
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes correctly."""
        assert self.preprocessor is not None
        assert hasattr(self.preprocessor, 'lemmatizer')
        assert hasattr(self.preprocessor, 'stopwords_set')
    
    def test_preprocess_empty_text(self):
        """Test preprocessing empty text."""
        result = self.preprocessor.preprocess("")
        assert result == ""
    
    def test_preprocess_with_punctuation(self):
        """Test removing punctuation."""
        text = "Hello, World! This is a test."
        result = self.preprocessor.preprocess(text)
        
        # Should not contain punctuation
        assert ',' not in result
        assert '!' not in result
        assert '.' not in result
    
    def test_preprocess_lowercase(self):
        """Test lowercasing."""
        text = "Python Machine Learning"
        result = self.preprocessor.preprocess(text)
        
        # Should be lowercase
        assert result.islower() or result == ""
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "Python programming and machine learning"
        tokens = self.preprocessor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        texts = ["First text", "Second text", "Third text"]
        results = self.preprocessor.preprocess_batch(texts)
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
    
    def test_get_stats(self):
        """Test getting text statistics."""
        text = "This is a sample text for testing purposes."
        stats = self.preprocessor.get_stats(text)
        
        assert 'original_length' in stats
        assert 'original_tokens' in stats
        assert 'preprocessed_length' in stats
        assert 'preprocessed_tokens' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
