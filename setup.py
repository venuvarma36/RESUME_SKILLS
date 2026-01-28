"""
Setup script for Resume Skill Recognition System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="resume-skill-recognition",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready NLP and ML system for automated skill extraction and resume-JD matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/resume-skill-recognition",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "ocr": [
            "pdf2image>=1.16.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "resume-match=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.json"],
    },
    keywords=[
        "nlp", "machine-learning", "resume-parsing", "skill-extraction",
        "job-matching", "bert", "transformers", "recruitment", "hr-tech"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/resume-skill-recognition/issues",
        "Source": "https://github.com/yourusername/resume-skill-recognition",
    },
)
