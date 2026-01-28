# Resume Skill Recognition System - Complete Documentation

**Version:** 1.0.0  
**Last Updated:** January 26, 2026  
**Author:** AI Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Core Modules](#core-modules)
5. [Machine Learning Models](#machine-learning-models)
6. [Data Flow](#data-flow)
7. [Installation Guide](#installation-guide)
8. [Configuration](#configuration)
9. [Usage Guide](#usage-guide)
10. [API Reference](#api-reference)
11. [Web UI Guide](#web-ui-guide)
12. [Performance Optimization](#performance-optimization)
13. [Extending the System](#extending-the-system)
14. [Troubleshooting](#troubleshooting)
15. [Best Practices](#best-practices)
16. [FAQ](#faq)

---

## Overview

### What is Resume Skill Recognition System?

The Resume Skill Recognition System is an end-to-end, production-ready application that uses Natural Language Processing (NLP) and Machine Learning (ML) to automatically extract, classify, and match skills from resumes against job descriptions. The system provides intelligent candidate ranking and comprehensive analysis tools.

### Key Features

- **Multi-format Support**: PDF, DOCX, and scanned documents (via OCR)
- **Hybrid Skill Extraction**: Combines NER models and rule-based matching
- **Advanced Embeddings**: Uses BERT-based sentence transformers
- **Intelligent Matching**: Weighted category scoring with cosine similarity
- **Interactive Web UI**: Built with Streamlit and Plotly visualizations
- **Batch Processing**: Handle multiple resumes simultaneously
- **Export Capabilities**: CSV, JSON, and ZIP downloads
- **Production-Ready**: Comprehensive logging, error handling, and testing

### Use Cases

1. **HR Recruitment**: Automate initial resume screening
2. **Talent Acquisition**: Rank candidates by skill match
3. **Skill Gap Analysis**: Identify missing skills in candidates
4. **Resume Parsing**: Extract structured data from unstructured documents
5. **Job Matching**: Match resumes to multiple job descriptions

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Web UI          │  │  CLI Interface   │  │  Python API  │  │
│  │  (Streamlit)     │  │  (Argparse)      │  │  (Module)    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MATCHING ENGINE                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ResumeJDMatcher (Orchestrator)                           │  │
│  │  - Coordinates all modules                                │  │
│  │  - Implements matching algorithm                          │  │
│  │  - Generates results and reports                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ TEXT EXTRACTION  │  │ SKILL EXTRACTION │  │ FEATURE ENGINE   │
│                  │  │                  │  │                  │
│ - PDF Parser     │  │ - NER Model      │  │ - BERT Embeddings│
│ - DOCX Parser    │  │ - Rule-based     │  │ - Caching        │
│ - OCR (Tesseract)│  │ - Dictionary     │  │ - Normalization  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
                    ┌──────────────────┐
                    │  PREPROCESSING   │
                    │                  │
                    │ - Tokenization   │
                    │ - Normalization  │
                    │ - Stopword Remove│
                    │ - Lemmatization  │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  ML CLASSIFIER   │
                    │                  │
                    │ - SVM            │
                    │ - Random Forest  │
                    │ - Logistic Reg.  │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  UTILITIES       │
                    │                  │
                    │ - Config Manager │
                    │ - Logger         │
                    │ - Helpers        │
                    └──────────────────┘
```

### Design Patterns

1. **Singleton Pattern**: ConfigManager, LoggerManager (single global instance)
2. **Pipeline Pattern**: Sequential processing (extract → preprocess → extract skills → match)
3. **Strategy Pattern**: Multiple extraction methods (PDF/DOCX/OCR) and ML algorithms
4. **Factory Pattern**: Model initialization and loading
5. **Observer Pattern**: Logging and monitoring throughout the pipeline

### Modular Structure

```
skill_recognition/
├── config/                 # Configuration files
├── text_extraction/        # Document parsing
├── preprocessing/          # Text normalization
├── skill_extraction/       # Skill identification
├── feature_engineering/    # Embedding generation
├── ml_model/              # Classification models
├── matching_engine/        # Core matching logic
├── ui/                    # Web interface
├── utils/                 # Shared utilities
└── tests/                 # Unit tests
```

---

## Technology Stack

### Core Technologies

#### Natural Language Processing
- **NLTK (Natural Language Toolkit)**: Tokenization, stopword removal, lemmatization
- **spaCy**: Advanced NLP pipeline (optional)
- **Transformers (Hugging Face)**: BERT-based models for NER and embeddings

#### Machine Learning
- **PyTorch**: Deep learning framework for BERT models
- **scikit-learn**: Traditional ML algorithms (SVM, Random Forest, Logistic Regression)
- **SentenceTransformers**: Semantic embeddings using `all-MiniLM-L6-v2`

#### Document Processing
- **PyPDF2**: PDF text extraction
- **pdfplumber**: Advanced PDF parsing with layout detection
- **python-docx**: Microsoft Word (DOCX) document parsing
- **pytesseract**: OCR for scanned documents
- **pdf2image**: Convert PDF pages to images for OCR

#### Data Processing
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing (distance metrics, optimization)

#### Visualization & UI
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations (charts, graphs, heatmaps)
- **Matplotlib**: Static plotting (optional)

#### Configuration & Utilities
- **PyYAML**: YAML configuration file parsing
- **colorlog**: Colored console logging
- **tqdm**: Progress bars

#### Testing & Development
- **pytest**: Unit testing framework
- **pytest-cov**: Code coverage reports

### Hardware Requirements

#### Minimum (CPU-Only)
- **CPU**: 2+ cores, 2.0 GHz+
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

#### Recommended
- **CPU**: 4+ cores, 3.0 GHz+
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (optional, 2-3x faster)
- **Storage**: 10 GB SSD

#### Performance Notes
- **CPU Performance**: 2-5 seconds per resume
- **GPU Performance**: 0.5-1 second per resume (3-5x speedup)
- **First Run**: 1-2 GB model download (one-time)

---

## Core Modules

### 1. Text Extraction Module

**Location**: `text_extraction/text_extractor.py`

**Purpose**: Extract text from various document formats with automatic fallback to OCR.

#### Class: `TextExtractor`

**Features**:
- Multi-format support (PDF, DOCX, TXT, scanned PDFs)
- Automatic OCR fallback for insufficient text extraction
- Batch processing capabilities
- Error handling and logging

**Methods**:

```python
class TextExtractor:
    def extract(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from a file.
        
        Returns:
            {
                'success': bool,
                'text': str,
                'extraction_method': str,
                'error': str (if failed)
            }
        """
        
    def extract_batch(self, file_paths: List[str]) -> List[Dict]:
        """Extract text from multiple files."""
        
    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 and pdfplumber."""
        
    def extract_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX files."""
        
    def extract_with_ocr(self, file_path: str) -> str:
        """Extract text using Tesseract OCR."""
```

**Algorithm**:

1. **Detect File Format**: Check file extension
2. **Primary Extraction**: Use format-specific parser
   - PDF: Try PyPDF2 → pdfplumber
   - DOCX: Use python-docx
3. **Quality Check**: Verify extracted text length (> 100 chars)
4. **OCR Fallback**: If text insufficient, apply OCR
5. **Return Result**: Text + metadata

**Configuration** (`config/config.yaml`):

```yaml
text_extraction:
  pdf_parser: 'pdfplumber'  # or 'pypdf2'
  use_ocr_fallback: true
  min_text_length: 100
  ocr_language: 'eng'
```

---

### 2. Preprocessing Module

**Location**: `preprocessing/text_preprocessor.py`

**Purpose**: Normalize and clean text for downstream processing.

#### Class: `TextPreprocessor`

**Features**:
- Unicode normalization (NFKD)
- Case normalization
- Punctuation handling
- Stopword removal
- Lemmatization
- Technical term preservation (e.g., C++, Node.js)

**Methods**:

```python
class TextPreprocessor:
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        
        Steps:
        1. Normalize unicode
        2. Remove extra whitespace
        3. Preserve technical terms
        4. Lowercase
        5. Remove punctuation (except in technical terms)
        6. Remove stopwords
        7. Lemmatize
        """
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords."""
        
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduce words to base form."""
```

**Technical Term Preservation**:

The preprocessor preserves common programming patterns:
- Languages: `C++`, `C#`, `Objective-C`
- Frameworks: `Node.js`, `Vue.js`, `Next.js`
- Protocols: `HTTP/2`, `IPv6`
- Versions: `Python3.10`, `Java11`

**Configuration**:

```yaml
preprocessing:
  lowercase: true
  remove_stopwords: true
  remove_punctuation: true
  lemmatize: true
  preserve_technical_terms: true
```

---

### 3. Skill Extraction Module

**Location**: `skill_extraction/skill_extractor.py`

**Purpose**: Extract skills from text using hybrid NER + rule-based approach.

#### Class: `SkillExtractor`

**Features**:
- NER-based extraction using BERT models
- Rule-based dictionary matching
- Synonym normalization
- Deduplication
- Skill validation and filtering

**Methods**:

```python
class SkillExtractor:
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text.
        
        Returns:
            {
                'technical_skills': List[str],
                'tools': List[str],
                'frameworks': List[str],
                'soft_skills': List[str]
            }
        """
        
    def _extract_with_ner(self, text: str) -> Dict:
        """Use NER model to extract skills."""
        
    def _extract_with_rules(self, text: str) -> Dict:
        """Use regex and dictionary matching."""
        
    def _is_valid_skill(self, skill: str) -> bool:
        """Validate extracted skill."""
```

**Hybrid Extraction Strategy**:

1. **NER Extraction**:
   - Model: `dslim/bert-base-NER` (default)
   - Confidence threshold: 0.7
   - Entity types: ORG, MISC
   
2. **Rule-Based Extraction**:
   - Dictionary: 500+ predefined skills
   - Case-insensitive matching
   - Word boundary detection
   - Pattern: `\b{skill}\b`

3. **Skill Validation**:
   - Minimum length: 3 characters (except whitelisted 2-char skills: C, R, Go, AI, ML, CI, CD, JS, TS)
   - No special character prefixes (#, $, @)
   - Must contain letters
   - Not pure numbers
   
4. **Merge & Deduplicate**:
   - Combine NER and rule-based results
   - Normalize synonyms (e.g., "Javascript" → "JavaScript")
   - Remove duplicates (case-insensitive)

**Skill Categories**:

```yaml
skills_dictionary:
  technical_skills:
    - Python
    - Java
    - C++
    - Machine Learning
    - Deep Learning
    # ... 200+ skills
    
  tools:
    - Git
    - Docker
    - Kubernetes
    # ... 150+ tools
    
  frameworks:
    - React
    - Django
    - TensorFlow
    # ... 100+ frameworks
    
  soft_skills:
    - Communication
    - Leadership
    - Problem Solving
    # ... 50+ soft skills
```

**Configuration**:

```yaml
skill_extraction:
  use_ner_model: true
  use_rule_based: true
  ner_model_name: 'dslim/bert-base-NER'
  confidence_threshold: 0.7
  deduplicate: true
  normalize_synonyms: true
```

---

### 4. Feature Engineering Module

**Location**: `feature_engineering/feature_engineer.py`

**Purpose**: Generate semantic embeddings for skill matching.

#### Class: `FeatureEngineer`

**Features**:
- BERT-based sentence embeddings
- Weighted skill embeddings by category
- Embedding caching for performance
- L2 normalization

**Methods**:

```python
class FeatureEngineer:
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate 384-dimensional embedding.
        
        Model: sentence-transformers/all-MiniLM-L6-v2
        """
        
    def generate_weighted_skill_embedding(
        self, skills: Dict[str, List[str]]
    ) -> np.ndarray:
        """
        Generate weighted embedding from categorized skills.
        
        Weights:
        - technical_skills: 0.5
        - tools: 0.3
        - frameworks: 0.15
        - soft_skills: 0.05
        """
        
    def generate_batch_embeddings(
        self, texts: List[str]
    ) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently."""
```

**Embedding Model**:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Performance**: ~3000 sentences/sec on GPU, ~500 on CPU
- **Size**: ~80 MB

**Weighted Embedding Formula**:

```
E_weighted = Σ(w_i × E_category_i)

Where:
- w_technical = 0.5
- w_tools = 0.3
- w_frameworks = 0.15
- w_soft = 0.05
- Σw_i = 1.0
```

**Configuration**:

```yaml
feature_engineering:
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'
  embedding_dim: 384
  normalize_embeddings: true
  use_cache: true
  cache_size: 1000
```

---

### 5. ML Model Module

**Location**: `ml_model/classifier.py`

**Purpose**: Train and use classifiers for skill categorization.

#### Class: `SkillClassifier`

**Supported Algorithms**:
1. **SVM (Support Vector Machine)** - Default
2. **Random Forest**
3. **Logistic Regression**

**Methods**:

```python
class SkillClassifier:
    def __init__(self, algorithm: str = 'svm'):
        """Initialize classifier with algorithm choice."""
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict skill categories."""
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Returns:
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'confusion_matrix': np.ndarray
            }
        """
        
    def save_model(self, path: str):
        """Save trained model to disk."""
        
    def load_model(self, path: str):
        """Load pre-trained model."""
```

**SVM Configuration**:

```python
SVC(
    kernel='linear',      # Linear kernel for high-dimensional data
    C=1.0,               # Regularization parameter
    probability=True,     # Enable probability estimates
    class_weight='balanced'  # Handle class imbalance
)
```

**Random Forest Configuration**:

```python
RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=None,      # No depth limit
    min_samples_split=2,
    class_weight='balanced'
)
```

**Configuration**:

```yaml
ml_model:
  algorithm: 'svm'  # 'svm', 'random_forest', 'logistic_regression'
  model_path: 'models/skill_classifier.pkl'
  
  svm:
    kernel: 'linear'
    C: 1.0
    
  random_forest:
    n_estimators: 100
    max_depth: null
```

---

### 6. Matching Engine

**Location**: `matching_engine/matcher.py`

**Purpose**: Core orchestrator for resume-JD matching.

#### Class: `ResumeJDMatcher`

**Methods**:

```python
class ResumeJDMatcher:
    def process_resume(self, file_path: str) -> Dict:
        """
        Process a single resume.
        
        Pipeline:
        1. Extract text
        2. Preprocess
        3. Extract skills
        4. Generate embeddings
        
        Returns resume data dict.
        """
        
    def process_job_description(self, jd_text: str) -> Dict:
        """Process job description (same pipeline as resume)."""
        
    def compute_match_score(
        self, resume_data: Dict, jd_data: Dict
    ) -> Dict:
        """
        Compute comprehensive match score.
        
        Returns:
            {
                'overall_score': float,
                'match_percentage': str,
                'category_scores': Dict[str, float],
                'embedding_similarity': float,
                'matched_skills': List[str],
                'missing_skills': List[str]
            }
        """
        
    def match_resumes_to_jd(
        self, resume_paths: List[str], jd_text: str
    ) -> pd.DataFrame:
        """
        Match multiple resumes to a job description.
        
        Returns sorted DataFrame with rankings.
        """
```

**Matching Algorithm**:

```python
# 1. Category-wise Matching (Jaccard Similarity)
for category in ['technical_skills', 'tools', 'frameworks', 'soft_skills']:
    resume_skills = set(resume_data['skills'][category])
    jd_skills = set(jd_data['skills'][category])
    
    intersection = resume_skills & jd_skills
    union = resume_skills | jd_skills
    
    category_score = len(intersection) / len(union) if union else 0
    category_scores[category] = category_score

# 2. Weighted Category Score
weighted_score = (
    0.50 * category_scores['technical_skills'] +
    0.30 * category_scores['tools'] +
    0.15 * category_scores['frameworks'] +
    0.05 * category_scores['soft_skills']
)

# 3. Embedding Similarity (Cosine)
embedding_similarity = cosine_similarity(
    resume_embedding, jd_embedding
)

# 4. Overall Score (Hybrid)
overall_score = (
    0.60 * weighted_score +
    0.40 * embedding_similarity
)
```

**Scoring Breakdown**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Category Matching | 60% | Exact skill matches by category |
| Embedding Similarity | 40% | Semantic similarity of full text |
| Technical Skills | 50% | Within category matching |
| Tools | 30% | Within category matching |
| Frameworks | 15% | Within category matching |
| Soft Skills | 5% | Within category matching |

**Configuration**:

```yaml
matching:
  similarity_metric: 'cosine'
  
  weights:
    technical_skills: 0.5
    tools: 0.3
    frameworks: 0.15
    soft_skills: 0.05
    
  hybrid_weights:
    category_matching: 0.6
    embedding_similarity: 0.4
```

---

## Machine Learning Models

### 1. Named Entity Recognition (NER)

**Model**: `dslim/bert-base-NER`

**Type**: BERT-based token classification

**Purpose**: Identify skill entities in text

**Architecture**:
- Base: BERT-base (12 layers, 768 hidden units)
- Task: Token classification
- Labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

**Training Data**: CoNLL-2003 (news articles)

**Performance**:
- F1 Score: ~91% on CoNLL-2003
- Inference: ~50ms per resume (CPU)

**Usage**:
```python
from transformers import pipeline

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

entities = ner("I have experience with Python and TensorFlow")
# [{'word': 'Python', 'entity_group': 'MISC', 'score': 0.95}, ...]
```

**Alternative Models**:
- `dbmdz/bert-large-cased-finetuned-conll03-english`
- `xlm-roberta-large-finetuned-conll03-english`

---

### 2. Sentence Embeddings

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Type**: Sentence-BERT (SBERT)

**Purpose**: Generate semantic embeddings of text

**Architecture**:
- Base: MiniLM-L6 (6 layers, 384 hidden units)
- Pooling: Mean pooling
- Output: 384-dimensional dense vectors

**Training**:
- Method: Contrastive learning (sentence pairs)
- Data: 1B+ sentence pairs
- Objective: Maximize similarity of semantically similar sentences

**Performance**:
- Speed: ~3000 sentences/sec (GPU), ~500 (CPU)
- Quality: 68.0% on STS benchmark
- Size: 80 MB

**Usage**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([
    "Python developer with ML experience",
    "Machine learning engineer skilled in Python"
])

similarity = cosine_similarity(embeddings[0], embeddings[1])
# 0.87 (high similarity)
```

**Alternative Models**:
- `all-mpnet-base-v2` (768-dim, slower, more accurate)
- `paraphrase-MiniLM-L3-v2` (384-dim, faster, less accurate)

---

### 3. Skill Classifier

**Algorithms**: SVM, Random Forest, Logistic Regression

**Purpose**: Categorize skills into technical/tools/frameworks/soft

**Default**: SVM with linear kernel

**SVM Details**:
- **Kernel**: Linear (best for high-dimensional sparse data)
- **C parameter**: 1.0 (regularization strength)
- **Multi-class**: One-vs-Rest
- **Features**: TF-IDF or embeddings

**Training Process**:
```python
# 1. Prepare training data
X = embeddings_of_skills  # (n_samples, 384)
y = skill_categories       # (n_samples,)

# 2. Train classifier
clf = SVC(kernel='linear', C=1.0, probability=True)
clf.fit(X, y)

# 3. Evaluate
accuracy = clf.score(X_test, y_test)

# 4. Save
joblib.dump(clf, 'skill_classifier.pkl')
```

**Performance**:
- Accuracy: ~85-90% (depends on training data)
- Inference: <1ms per skill
- Training time: ~30 seconds for 1000 samples

---

## Data Flow

### Resume Processing Pipeline

```
Input: Resume PDF/DOCX
         ↓
┌─────────────────────┐
│ 1. TEXT EXTRACTION │
│   - Detect format   │
│   - Parse document  │
│   - OCR if needed   │
└──────────┬──────────┘
           ↓
        raw_text
           ↓
┌─────────────────────┐
│ 2. PREPROCESSING   │
│   - Normalize       │
│   - Tokenize        │
│   - Lemmatize       │
└──────────┬──────────┘
           ↓
    cleaned_text
           ↓
┌─────────────────────┐
│ 3. SKILL EXTRACTION│
│   - NER model       │
│   - Rule matching   │
│   - Validation      │
└──────────┬──────────┘
           ↓
categorized_skills = {
    'technical_skills': [...],
    'tools': [...],
    'frameworks': [...],
    'soft_skills': [...]
}
           ↓
┌─────────────────────┐
│ 4. FEATURE GEN     │
│   - Generate embed. │
│   - Weight by cat.  │
│   - Normalize       │
└──────────┬──────────┘
           ↓
embedding_vector (384-dim)
           ↓
        OUTPUT:
resume_data = {
    'text': str,
    'skills': Dict[category, List[str]],
    'embedding': np.ndarray,
    'success': bool
}
```

### Matching Pipeline

```
Input: resume_data, jd_data
         ↓
┌─────────────────────────────┐
│ 1. CATEGORY MATCHING        │
│   For each category:        │
│   - Intersection            │
│   - Union                   │
│   - Jaccard = |A∩B| / |A∪B| │
└──────────┬──────────────────┘
           ↓
category_scores = {
    'technical_skills': 0.75,
    'tools': 0.60,
    ...
}
           ↓
┌─────────────────────────────┐
│ 2. WEIGHTED CATEGORY SCORE  │
│   score = Σ(weight_i × cat_i)│
└──────────┬──────────────────┘
           ↓
weighted_category_score = 0.68
           ↓
┌─────────────────────────────┐
│ 3. EMBEDDING SIMILARITY     │
│   cosine_sim(emb_r, emb_jd) │
└──────────┬──────────────────┘
           ↓
embedding_similarity = 0.72
           ↓
┌─────────────────────────────┐
│ 4. HYBRID SCORE             │
│   0.6×category + 0.4×embed  │
└──────────┬──────────────────┘
           ↓
overall_score = 0.70 (70%)
           ↓
        OUTPUT:
match_result = {
    'overall_score': 0.70,
    'match_percentage': '70.0%',
    'category_scores': {...},
    'embedding_similarity': 0.72,
    'matched_skills': [...],
    'missing_skills': [...]
}
```

---

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- 8 GB RAM minimum (16 GB recommended)
- 5 GB free disk space
- Internet connection (for initial model downloads)

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/your-org/resume-skill-recognition.git
cd resume-skill-recognition
```

#### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n resume-skill python=3.10
conda activate resume-skill
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

#### 5. Optional: Install Tesseract OCR

**Windows:**
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Run installer
- Add to PATH: `C:\Program Files\Tesseract-OCR`

**macOS:**
```bash
brew install tesseract
brew install poppler
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

#### 6. Verify Installation

```bash
python -c "from matching_engine import ResumeJDMatcher; print('Installation successful!')"
```

---

## Configuration

### Configuration File Structure

**Location**: `config/config.yaml`

```yaml
# Text Extraction
text_extraction:
  pdf_parser: 'pdfplumber'
  use_ocr_fallback: true
  min_text_length: 100
  ocr_language: 'eng'

# Preprocessing
preprocessing:
  lowercase: true
  remove_stopwords: true
  remove_punctuation: true
  lemmatize: true
  preserve_technical_terms: true

# Skill Extraction
skill_extraction:
  use_ner_model: true
  use_rule_based: true
  ner_model_name: 'dslim/bert-base-NER'
  confidence_threshold: 0.7
  deduplicate: true
  normalize_synonyms: true

# Feature Engineering
feature_engineering:
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'
  embedding_dim: 384
  normalize_embeddings: true
  use_cache: true
  cache_size: 1000

# Matching
matching:
  similarity_metric: 'cosine'
  weights:
    technical_skills: 0.5
    tools: 0.3
    frameworks: 0.15
    soft_skills: 0.05
  hybrid_weights:
    category_matching: 0.6
    embedding_similarity: 0.4

# ML Model
ml_model:
  algorithm: 'svm'
  model_path: 'models/skill_classifier.pkl'

# Logging
logging:
  level: 'INFO'
  log_to_file: true
  log_to_console: true
  log_file: 'logs/app.log'

# UI
ui:
  page_title: 'Resume Skill Recognition'
  page_icon: '📄'
  layout: 'wide'

# Paths
paths:
  data_dir: 'data'
  models_dir: 'models'
  logs_dir: 'logs'
  output_dir: 'output'
```

### Modifying Configuration

#### Option 1: Edit YAML File

```bash
nano config/config.yaml
```

#### Option 2: Programmatic Access

```python
from utils import config

# Get value
threshold = config.get('skill_extraction.confidence_threshold')

# Set value (runtime only)
config.set('logging.level', 'DEBUG')

# Get nested value with default
cache_size = config.get('feature_engineering.cache_size', default=500)
```

---

## Usage Guide

### Method 1: Web UI (Recommended)

```bash
python main.py --ui
```

Then open browser to: `http://localhost:8501`

**Steps**:
1. Upload resume files (PDF/DOCX)
2. Paste job description
3. Click "Match Resumes"
4. View results and download

---

### Method 2: Command Line Interface

```bash
# Match single resume to JD
python main.py --resume path/to/resume.pdf --jd path/to/job_description.txt

# Match multiple resumes
python main.py --resumes resume1.pdf resume2.pdf resume3.pdf --jd jd.txt

# Specify output format
python main.py --resume resume.pdf --jd jd.txt --output json --output-file results.json

# Enable debug logging
python main.py --resume resume.pdf --jd jd.txt --verbose
```

---

### Method 3: Python API

```python
from matching_engine import ResumeJDMatcher

# Initialize matcher
matcher = ResumeJDMatcher()

# Process single resume
resume_data = matcher.process_resume('path/to/resume.pdf')
print(f"Extracted {len(resume_data['skills'])} skill categories")

# Process job description
jd_data = matcher.process_job_description("Job description text here...")

# Compute match score
match_result = matcher.compute_match_score(resume_data, jd_data)
print(f"Match Score: {match_result['match_percentage']}")

# Batch processing
results_df = matcher.match_resumes_to_jd(
    resume_paths=['resume1.pdf', 'resume2.pdf'],
    jd_text="Job description..."
)

# Export results
results_df.to_csv('results.csv', index=False)
```

---

## API Reference

### ResumeJDMatcher

```python
class ResumeJDMatcher:
    """Main class for resume-JD matching."""
    
    def __init__(self):
        """Initialize all components."""
        
    def process_resume(self, file_path: str) -> Dict[str, any]:
        """
        Process a resume file.
        
        Args:
            file_path: Path to resume file (PDF/DOCX)
            
        Returns:
            {
                'success': bool,
                'text': str,
                'skills': {
                    'technical_skills': List[str],
                    'tools': List[str],
                    'frameworks': List[str],
                    'soft_skills': List[str]
                },
                'embedding': np.ndarray,
                'extraction_method': str
            }
        """
        
    def process_job_description(self, jd_text: str) -> Dict[str, any]:
        """
        Process a job description.
        
        Args:
            jd_text: Job description text
            
        Returns:
            Same structure as process_resume()
        """
        
    def compute_match_score(
        self, 
        resume_data: Dict, 
        jd_data: Dict
    ) -> Dict[str, any]:
        """
        Compute match score between resume and JD.
        
        Args:
            resume_data: Output from process_resume()
            jd_data: Output from process_job_description()
            
        Returns:
            {
                'overall_score': float,  # 0.0 to 1.0
                'match_percentage': str,  # "75.5%"
                'category_scores': {
                    'technical_skills': float,
                    'tools': float,
                    'frameworks': float,
                    'soft_skills': float
                },
                'embedding_similarity': float,
                'matched_skills': List[str],
                'missing_skills': List[str]
            }
        """
        
    def match_resumes_to_jd(
        self,
        resume_paths: List[str],
        jd_text: str
    ) -> pd.DataFrame:
        """
        Match multiple resumes to a job description.
        
        Args:
            resume_paths: List of resume file paths
            jd_text: Job description text
            
        Returns:
            DataFrame with columns:
            - rank: int
            - resume_file: str
            - overall_score: float
            - match_percentage: str
            - technical_skills_score: float
            - tools_score: float
            - frameworks_score: float
            - soft_skills_score: float
            - matched_skills_count: int
            - missing_skills_count: int
            - matched_skills: str (comma-separated)
            - missing_skills: str (comma-separated)
            - all_extracted_skills: str (comma-separated)
            
        Sorted by overall_score descending.
        """
```

---

## Web UI Guide

### Main Interface

#### 1. Sidebar
- **File Upload**: Drag-and-drop or browse for resumes
- **Job Description**: Text area for JD input
- **Match Button**: Start processing
- **Export Results**: Download CSV/JSON

#### 2. Main Area
- **Matching Summary**: 4 key metrics
- **Skills Overview**: All candidates with expandable sections
  - All extracted skills
  - Matched skills
  - Missing skills
  - Individual download button
- **Download Section**: 
  - Multi-select dropdown
  - Download selected resumes
  - Download qualified candidates button
- **Detailed Analysis** (expandable):
  - Score distribution charts
  - Category-wise heatmap
  - Detailed results table
  - Individual resume analysis

### Features

#### Interactive Visualizations
- **Bar Chart**: Top 10 candidates
- **Histogram**: Score distribution
- **Heatmap**: Category performance
- **Gauge Chart**: Overall match score
- **Radar Chart**: Category breakdown

#### Export Options
- **CSV**: Tabular results
- **JSON**: Structured data
- **ZIP**: Selected resume files

#### Responsive Design
- Wide layout for dashboards
- Expandable sections to reduce clutter
- Color-coded skill pills
- Real-time updates

---

## Performance Optimization

### 1. Model Loading

**Problem**: Models take time to load on first use.

**Solution**: Pre-load models during initialization.

```python
# Lazy loading (default)
matcher = ResumeJDMatcher()  # Fast init, slow first use

# Pre-loading (recommended for production)
matcher = ResumeJDMatcher()
matcher.skill_extractor._initialize_ner_model()  # Warm-up
```

### 2. Batch Processing

**Problem**: Processing resumes one-by-one is slow.

**Solution**: Use batch operations where possible.

```python
# Slow
embeddings = [model.encode(text) for text in texts]

# Fast (3-5x speedup)
embeddings = model.encode(texts, batch_size=32)
```

### 3. Embedding Caching

**Problem**: Same texts are embedded multiple times.

**Solution**: Enable caching in configuration.

```yaml
feature_engineering:
  use_cache: true
  cache_size: 1000  # LRU cache
```

### 4. GPU Acceleration

**Problem**: CPU inference is slow.

**Solution**: Use GPU if available (automatic detection).

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = 0  # GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1  # CPU
    print("Using CPU")
```

**Performance Gains**:
- NER: 3-5x faster
- Embeddings: 5-10x faster
- Overall: 3-5x faster

### 5. Reduce Model Size

**Option**: Use smaller/faster models.

```yaml
skill_extraction:
  ner_model_name: 'dslim/bert-base-NER'  # Default
  # ner_model_name: 'distilbert-base-NER'  # Faster, less accurate

feature_engineering:
  embedding_model: 'all-MiniLM-L6-v2'  # Default
  # embedding_model: 'paraphrase-MiniLM-L3-v2'  # 2x faster, less accurate
```

### 6. Parallel Processing

**For large batches**: Use multiprocessing.

```python
from concurrent.futures import ProcessPoolExecutor

def process_resume(path):
    return matcher.process_resume(path)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_resume, resume_paths))
```

---

## Extending the System

### Adding New Skill Categories

**1. Update Dictionary** (`config/skills_dictionary.json`):

```json
{
  "technical_skills": [...],
  "tools": [...],
  "frameworks": [...],
  "soft_skills": [...],
  "certifications": [
    "AWS Certified",
    "PMP",
    "CISSP"
  ]
}
```

**2. Update Weights** (`config/config.yaml`):

```yaml
matching:
  weights:
    technical_skills: 0.4
    tools: 0.25
    frameworks: 0.15
    soft_skills: 0.05
    certifications: 0.15
```

**3. Update Code** (if needed):

```python
# In skill_extractor.py
def _empty_result(self):
    return {
        'technical_skills': [],
        'tools': [],
        'frameworks': [],
        'soft_skills': [],
        'certifications': []  # New category
    }
```

### Using a Different NER Model

**1. Install model**:

```bash
pip install transformers
```

**2. Update config**:

```yaml
skill_extraction:
  ner_model_name: 'your-org/your-ner-model'
```

**3. Fine-tune for your domain** (optional):

```python
from transformers import AutoModelForTokenClassification, Trainer

model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)
trainer = Trainer(model=model, train_dataset=your_dataset, ...)
trainer.train()
model.save_pretrained('models/custom-ner')
```

### Adding New Export Formats

**Example: Add Excel export**:

```python
# In ui/app.py
import openpyxl

# Create Excel export
with pd.ExcelWriter('results.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Results', index=False)

st.download_button(
    label="📊 Excel",
    data=open('results.xlsx', 'rb'),
    file_name='results.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
```

---

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Error**: `ConnectionError: Could not download model`

**Solutions**:
- Check internet connection
- Use VPN if behind firewall
- Manually download model:
  ```python
  from transformers import AutoModel
  AutoModel.from_pretrained('dslim/bert-base-NER', cache_dir='./models')
  ```

#### 2. Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size
- Use CPU instead of GPU
- Process resumes one at a time
- Use smaller model

#### 3. OCR Not Working

**Error**: `TesseractNotFoundError`

**Solutions**:
- Install Tesseract: See installation guide
- Add to PATH (Windows)
- Specify path in config:
  ```yaml
  text_extraction:
    tesseract_cmd: 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
  ```

#### 4. Slow Performance

**Symptoms**: High processing time

**Solutions**:
- Enable GPU if available
- Enable embedding cache
- Use smaller models
- Process in batches

#### 5. Poor Matching Results

**Symptoms**: Low scores or irrelevant matches

**Solutions**:
- Expand skill dictionary
- Adjust category weights
- Lower confidence threshold
- Add domain-specific skills

---

## Best Practices

### 1. Resume Quality

- **Prefer machine-readable formats**: PDF (text-based), DOCX
- **Avoid scanned images**: Use OCR preprocessing if necessary
- **Well-formatted resumes**: Clear sections, proper headings

### 2. Job Description Quality

- **Be specific**: List exact skills, tools, frameworks
- **Use standard terms**: "Python" not "python programming"
- **Include all requirements**: Hard skills and soft skills
- **Avoid ambiguity**: "React.js" not "React experience"

### 3. Configuration Tuning

- **Category weights**: Adjust based on job importance
- **Confidence threshold**: Lower for more skills, higher for precision
- **Embedding model**: Balance speed vs accuracy

### 4. Production Deployment

- **Pre-load models**: Faster first response
- **Enable caching**: Reduce redundant computation
- **Use GPU**: 3-5x speedup
- **Monitor logs**: Track errors and performance
- **Set up CI/CD**: Automated testing

### 5. Data Privacy

- **Secure file uploads**: Use HTTPS
- **Temporary storage**: Delete files after processing
- **No persistent storage**: Don't save resumes to disk
- **Anonymize data**: Remove PII before analysis

---

## FAQ

### General

**Q: Does this system require internet access?**  
A: Yes, for initial model downloads (~1-2 GB). After that, it works offline.

**Q: Can I use this commercially?**  
A: Check licenses of individual models. Most are open-source (Apache 2.0, MIT).

**Q: What file formats are supported?**  
A: PDF, DOCX, TXT, and scanned PDFs (with OCR).

### Technical

**Q: Do I need a GPU?**  
A: No, system runs on CPU. GPU provides 3-5x speedup but is optional.

**Q: How accurate is the skill extraction?**  
A: ~85-90% precision for common skills. Fine-tuning improves accuracy.

**Q: Can I add custom skills?**  
A: Yes, edit `config/skills_dictionary.json`.

**Q: How are skills categorized?**  
A: Using predefined dictionary + ML classifier. You can customize categories.

**Q: What's the maximum resume size?**  
A: No hard limit, but performance degrades for 50+ page documents.

### Performance

**Q: How fast is the system?**  
A: 2-5 seconds per resume on CPU, 0.5-1 second on GPU.

**Q: Can I process 1000 resumes at once?**  
A: Yes, but consider batch processing and parallel execution.

**Q: Does it cache results?**  
A: Embeddings are cached. Full results are not cached by default.

### Deployment

**Q: Can I deploy to cloud?**  
A: Yes, supports Docker, AWS, Azure, GCP. See deployment guides.

**Q: Is there a REST API?**  
A: Not included, but easy to add with FastAPI or Flask.

**Q: Can multiple users access simultaneously?**  
A: Web UI supports multiple concurrent users with session state.

---

## Appendix

### A. Dependencies

**Core** (`requirements.txt`):
```
nltk==3.8.1
spacy==3.5.3
transformers==4.30.2
torch==2.0.1
sentence-transformers==2.2.2
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
scipy==1.11.1
streamlit==1.24.0
plotly==5.15.0
PyYAML==6.0
PyPDF2==3.0.1
pdfplumber==0.9.0
python-docx==0.8.11
pytesseract==0.3.10
pdf2image==1.16.3
colorlog==6.7.0
tqdm==4.65.0
pytest==7.4.0
pytest-cov==4.1.0
```

### B. File Structure

```
skill_recognition/
├── config/
│   ├── config.yaml
│   └── skills_dictionary.json
├── text_extraction/
│   ├── __init__.py
│   └── text_extractor.py
├── preprocessing/
│   ├── __init__.py
│   └── text_preprocessor.py
├── skill_extraction/
│   ├── __init__.py
│   └── skill_extractor.py
├── feature_engineering/
│   ├── __init__.py
│   └── feature_engineer.py
├── ml_model/
│   ├── __init__.py
│   └── classifier.py
├── matching_engine/
│   ├── __init__.py
│   └── matcher.py
├── ui/
│   ├── __init__.py
│   └── app.py
├── utils/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── logger.py
│   └── helpers.py
├── tests/
│   ├── test_extraction.py
│   ├── test_preprocessing.py
│   ├── test_skill_extraction.py
│   └── test_utils.py
├── data/
├── models/
├── logs/
├── output/
├── main.py
├── examples.py
├── requirements.txt
├── README.md
├── INSTALLATION.md
├── QUICKSTART.md
├── DOCUMENTATION.md
├── FAQ.md
└── LICENSE
```

### C. References

**Papers**:
1. BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
2. Sentence-BERT: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
3. Named Entity Recognition: "Named Entity Recognition with Bidirectional LSTM-CNNs" (Chiu & Nichols, 2016)

**Frameworks**:
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Sentence Transformers: https://www.sbert.net
- NLTK: https://www.nltk.org
- Streamlit: https://docs.streamlit.io

**Models**:
- BERT-base-NER: https://huggingface.co/dslim/bert-base-NER
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

## Changelog

### Version 1.0.0 (January 26, 2026)

**Initial Release**:
- ✅ Multi-format text extraction (PDF, DOCX, OCR)
- ✅ Hybrid skill extraction (NER + rule-based)
- ✅ BERT-based embeddings
- ✅ Weighted category matching
- ✅ Interactive web UI
- ✅ Batch processing
- ✅ Export to CSV/JSON/ZIP
- ✅ Comprehensive logging
- ✅ Unit tests
- ✅ Complete documentation

---

## Support & Contact

**Documentation**: See README.md, QUICKSTART.md, FAQ.md

**Issues**: Report bugs and feature requests on GitHub Issues

**Contributing**: See CONTRIBUTING.md for guidelines

**License**: MIT License - see LICENSE file

---

**End of Documentation**
