"""
Text Preprocessing Module for Resume Skill Recognition System
Handles cleaning, normalization, and tokenization of text data.
"""

import re
import string
import unicodedata
from typing import List, Optional, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import get_logger, config, normalize_text


logger = get_logger(__name__)


class TextPreprocessor:
    """Handles text preprocessing with configurable options."""
    
    def __init__(self, download_nltk_data: bool = True):
        """
        Initialize text preprocessor.
        
        Args:
            download_nltk_data: Whether to download required NLTK data
        """
        # Download required NLTK data
        if download_nltk_data:
            self._download_nltk_resources()
        
        # Load configuration
        self.lowercase = config.get('preprocessing.lowercase', True)
        self.remove_punctuation = config.get('preprocessing.remove_punctuation', True)
        self.remove_stopwords = config.get('preprocessing.remove_stopwords', True)
        self.lemmatize = config.get('preprocessing.lemmatize', True)
        self.preserve_technical = config.get('preprocessing.preserve_technical_terms', True)
        self.technical_pattern = config.get('preprocessing.technical_terms_pattern', '')
        self.min_token_length = config.get('preprocessing.min_token_length', 2)
        self.max_token_length = config.get('preprocessing.max_token_length', 50)
        
        # Initialize NLTK components
        try:
            self.stopwords_set = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            logger.warning("NLTK data not found. Downloading...")
            self._download_nltk_resources()
            self.stopwords_set = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        
        # Compile technical terms pattern
        if self.technical_pattern:
            self.technical_regex = re.compile(self.technical_pattern)
        else:
            self.technical_regex = None
        
        logger.info("TextPreprocessor initialized with lowercase=%s, lemmatize=%s",
                   self.lowercase, self.lemmatize)
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning("Failed to download NLTK resource '%s': %s", resource, str(e))
    
    def preprocess(self, text: str, preserve_case_tokens: Optional[List[str]] = None) -> str:
        """
        Preprocess text with all configured steps.
        
        Args:
            text: Input text
            preserve_case_tokens: List of tokens to preserve case for
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = self._normalize_unicode(text)
        
        # Normalize whitespace
        text = normalize_text(text)
        
        # Extract and preserve technical terms
        technical_terms = []
        if self.preserve_technical and self.technical_regex:
            technical_terms = self.technical_regex.findall(text)
            # Replace with placeholders
            for i, term in enumerate(technical_terms):
                text = text.replace(term, f" __TECH_{i}__ ")
        
        # Convert to lowercase (if enabled)
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation (if enabled)
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        tokens = self._filter_tokens(tokens)
        
        # Remove stopwords (if enabled)
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        
        # Lemmatize (if enabled)
        if self.lemmatize:
            tokens = self._lemmatize_tokens(tokens)
        
        # Restore technical terms
        result_tokens = []
        for token in tokens:
            if token.startswith('__tech_') and token.endswith('__'):
                # Extract index
                try:
                    idx = int(token.replace('__tech_', '').replace('__', ''))
                    if 0 <= idx < len(technical_terms):
                        result_tokens.append(technical_terms[idx])
                    else:
                        result_tokens.append(token)
                except ValueError:
                    result_tokens.append(token)
            else:
                result_tokens.append(token)
        
        return ' '.join(result_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        text = self.preprocess(text)
        return word_tokenize(text)
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize to NFKD form and encode to ASCII, ignoring errors
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without punctuation
        """
        # Keep alphanumeric and whitespace
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        return text
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by length and content.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        filtered = []
        for token in tokens:
            # Check length
            if len(token) < self.min_token_length or len(token) > self.max_token_length:
                continue
            
            # Check if token contains at least one alphanumeric character
            if not any(c.isalnum() for c in token):
                continue
            
            filtered.append(token)
        
        return filtered
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens without stopwords
        """
        return [token for token in tokens if token.lower() not in self.stopwords_set]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of preprocessed texts
        """
        logger.info("Preprocessing batch of %d texts", len(texts))
        return [self.preprocess(text) for text in texts]
    
    def get_stats(self, text: str) -> Dict[str, int]:
        """
        Get statistics about text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        tokens = word_tokenize(text)
        preprocessed = self.preprocess(text)
        preprocessed_tokens = word_tokenize(preprocessed)
        
        return {
            'original_length': len(text),
            'original_tokens': len(tokens),
            'preprocessed_length': len(preprocessed),
            'preprocessed_tokens': len(preprocessed_tokens),
            'reduction_ratio': 1 - (len(preprocessed_tokens) / max(len(tokens), 1))
        }


def preprocess_text(text: str, **kwargs) -> str:
    """
    Convenience function to preprocess text.
    
    Args:
        text: Input text
        **kwargs: Additional arguments for TextPreprocessor
        
    Returns:
        Preprocessed text
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)
