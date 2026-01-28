"""
Utility functions for Resume Skill Recognition System
Common helper functions used across the system.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np


def load_json(file_path: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: Indentation level
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and newlines.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def deduplicate_list(items: List[str], case_sensitive: bool = False) -> List[str]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        items: List of items
        case_sensitive: Whether to consider case
        
    Returns:
        Deduplicated list
    """
    if not case_sensitive:
        seen = set()
        result = []
        for item in items:
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                result.append(item)
        return result
    else:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def validate_file_type(file_path: str, allowed_extensions: List[str]) -> bool:
    """
    Validate if file has allowed extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.docx'])
        
    Returns:
        True if valid, False otherwise
    """
    file_ext = Path(file_path).suffix.lower()
    return file_ext in [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                       for ext in allowed_extensions]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def merge_skill_lists(*skill_lists: List[str]) -> List[str]:
    """
    Merge multiple skill lists, removing duplicates and normalizing.
    
    Args:
        skill_lists: Variable number of skill lists
        
    Returns:
        Merged and deduplicated skill list
    """
    all_skills = []
    for skill_list in skill_lists:
        if skill_list:
            all_skills.extend(skill_list)
    
    return deduplicate_list(all_skills, case_sensitive=False)


def calculate_overlap(list1: List[str], list2: List[str], 
                     case_sensitive: bool = False) -> Dict[str, Any]:
    """
    Calculate overlap between two lists.
    
    Args:
        list1: First list
        list2: Second list
        case_sensitive: Whether to consider case
        
    Returns:
        Dictionary with overlap statistics
    """
    if not case_sensitive:
        set1 = {item.lower() for item in list1}
        set2 = {item.lower() for item in list2}
    else:
        set1 = set(list1)
        set2 = set(list2)
    
    intersection = set1 & set2
    union = set1 | set2
    
    overlap_ratio = safe_divide(len(intersection), len(union))
    
    return {
        'common_items': list(intersection),
        'unique_to_list1': list(set1 - set2),
        'unique_to_list2': list(set2 - set1),
        'overlap_count': len(intersection),
        'overlap_ratio': overlap_ratio,
        'total_unique_items': len(union)
    }


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format float as percentage string.
    
    Args:
        value: Float value (0 to 1)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def truncate_text(text: str, max_length: int = 100, 
                  suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def ensure_list(value: Any) -> List:
    """
    Ensure value is a list.
    
    Args:
        value: Input value
        
    Returns:
        List representation of value
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (set, tuple)):
        return list(value)
    return [value]
