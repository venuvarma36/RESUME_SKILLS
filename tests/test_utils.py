"""
Unit tests for Utilities Module
"""

import pytest
import numpy as np
from utils import (
    normalize_text,
    deduplicate_list,
    cosine_similarity,
    validate_file_type,
    safe_divide,
    merge_skill_lists,
    calculate_overlap
)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "  Multiple   spaces   and\n\nnewlines  "
        result = normalize_text(text)
        
        assert "  " not in result
        assert result.startswith("Multiple")
        assert result.endswith("newlines")
    
    def test_deduplicate_list(self):
        """Test list deduplication."""
        items = ['apple', 'APPLE', 'banana', 'Apple', 'banana']
        result = deduplicate_list(items, case_sensitive=False)
        
        assert len(result) == 2  # apple and banana
        assert 'apple' in [item.lower() for item in result]
        assert 'banana' in [item.lower() for item in result]
    
    def test_deduplicate_list_case_sensitive(self):
        """Test case-sensitive deduplication."""
        items = ['apple', 'APPLE', 'Apple']
        result = deduplicate_list(items, case_sensitive=True)
        
        assert len(result) == 3
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])
        
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_validate_file_type(self):
        """Test file type validation."""
        assert validate_file_type("resume.pdf", ['.pdf', '.docx']) is True
        assert validate_file_type("resume.PDF", ['.pdf', '.docx']) is True
        assert validate_file_type("resume.txt", ['.pdf', '.docx']) is False
    
    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=99) == 99
    
    def test_merge_skill_lists(self):
        """Test merging skill lists."""
        list1 = ['Python', 'Java']
        list2 = ['Python', 'C++']
        list3 = ['JavaScript']
        
        merged = merge_skill_lists(list1, list2, list3)
        
        assert 'Python' in merged
        assert 'Java' in merged
        assert 'C++' in merged
        assert 'JavaScript' in merged
    
    def test_calculate_overlap(self):
        """Test overlap calculation."""
        list1 = ['Python', 'Java', 'C++']
        list2 = ['Python', 'JavaScript', 'C++']
        
        overlap = calculate_overlap(list1, list2)
        
        assert overlap['overlap_count'] == 2  # Python and C++
        assert 'python' in [s.lower() for s in overlap['common_items']]
        assert 'java' in [s.lower() for s in overlap['unique_to_list1']]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
