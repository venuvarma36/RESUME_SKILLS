"""
Data package for Resume Skill Recognition System
Contains sample data and utilities.
"""

from pathlib import Path

# Package info
__version__ = "1.0.0"

# Data directories
DATA_DIR = Path(__file__).parent
RESUMES_DIR = DATA_DIR / "resumes"
MODELS_DIR = DATA_DIR.parent / "models"
OUTPUT_DIR = DATA_DIR.parent / "output"

# Ensure directories exist
RESUMES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
