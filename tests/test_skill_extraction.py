"""
Unit tests for Skill Extraction Module
"""

import pytest
from skill_extraction import SkillExtractor


class TestSkillExtractor:
    """Test cases for SkillExtractor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = SkillExtractor()
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly."""
        assert self.extractor is not None
        assert self.extractor.skill_dict is not None
        assert 'technical_skills' in self.extractor.skill_dict
    
    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        result = self.extractor.extract("")
        
        assert isinstance(result, dict)
        assert 'technical_skills' in result
        assert 'tools' in result
        assert 'frameworks' in result
        assert 'soft_skills' in result
    
    def test_extract_with_known_skills(self):
        """Test extraction with known skills."""
        text = "I have experience with Python, Machine Learning, and TensorFlow."
        result = self.extractor.extract(text)
        
        # Should extract at least some skills
        all_skills = []
        for skills_list in result.values():
            all_skills.extend(skills_list)
        
        assert len(all_skills) > 0
    
    def test_extract_batch(self):
        """Test batch extraction."""
        texts = [
            "Python programming",
            "Java development",
            "Machine Learning expert"
        ]
        results = self.extractor.extract_batch(texts)
        
        assert isinstance(results, list)
        assert len(results) == len(texts)
    
    def test_deduplicate_skills(self):
        """Test skill deduplication."""
        skills = {
            'technical_skills': ['Python', 'python', 'PYTHON', 'Java'],
            'tools': ['Git', 'git']
        }
        
        result = self.extractor._deduplicate_skills(skills)
        
        # Python should appear only once
        assert len(result['technical_skills']) < len(skills['technical_skills'])
    
    def test_get_all_skills_flat(self):
        """Test flattening skills."""
        skills = {
            'technical_skills': ['Python', 'Java'],
            'tools': ['Git'],
            'frameworks': ['Django'],
            'soft_skills': []
        }
        
        flat_skills = self.extractor.get_all_skills_flat(skills)
        
        assert isinstance(flat_skills, list)
        assert len(flat_skills) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
