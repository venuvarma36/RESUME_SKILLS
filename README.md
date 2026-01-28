# Resume Skill Recognition & Matching System

> **Production-ready NLP and ML system for automated skill extraction and resume-job description matching**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This is a comprehensive, enterprise-grade system for automated skill recognition from resumes and intelligent matching with job descriptions. Built using state-of-the-art NLP techniques, transformer models, and machine learning algorithms.

### Key Features

- 📄 **Multi-format Support**: PDF (text & scanned), DOCX with automatic OCR fallback
- 🤖 **Hybrid Skill Extraction**: Combines transformer-based NER and rule-based methods
- 🎯 **Smart Matching**: Weighted similarity scoring with BERT embeddings
- 📊 **Rich Analytics**: Interactive visualizations and detailed reports
- 🔧 **Production Ready**: Robust error handling, logging, and configuration
- 🧪 **Fully Tested**: Comprehensive unit test coverage
- 🎨 **Modern UI**: Professional Streamlit web interface

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│  • PDF Files (text-based & scanned)                         │
│  • DOCX Files                                               │
│  • Job Description Text                                      │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              TEXT EXTRACTION LAYER                           │
│  • PyPDF2 / pdfplumber for PDF                             │
│  • python-docx for DOCX                                     │
│  • Automatic OCR fallback (pytesseract)                    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│           PREPROCESSING PIPELINE                             │
│  • Unicode normalization                                    │
│  • Lowercasing & punctuation removal                        │
│  • Stopword removal (NLTK)                                  │
│  • Lemmatization (WordNet)                                  │
│  • Technical term preservation                              │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│            SKILL EXTRACTION ENGINE                           │
│  • NER-based: BERT / DistilBERT                             │
│  • Rule-based: Regex + Dictionary matching                  │
│  • Synonym normalization                                    │
│  • Category classification (Technical, Tools, etc.)         │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│          FEATURE ENGINEERING                                 │
│  • BERT embeddings (Sentence Transformers)                  │
│  • Weighted skill vectors                                   │
│  • Embedding caching                                        │
│  • Vector normalization                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│        MATCHING ENGINE & ML MODEL                            │
│  • Cosine similarity computation                            │
│  • Category-wise scoring                                    │
│  • SVM classifier (optional)                                │
│  • Weighted aggregation                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              OUTPUT LAYER                                    │
│  • Ranked candidate list                                    │
│  • Match percentages                                        │
│  • Skill gap analysis                                       │
│  • Visualizations (charts, heatmaps)                        │
│  • Export (CSV, JSON)                                       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Skill_Recognition/
├── config/
│   ├── config.yaml              # System configuration
│   └── skills_dictionary.json   # Skill taxonomy
├── data/
│   ├── resumes/                 # Input resumes
│   └── models/                  # Trained models
├── text_extraction/
│   └── text_extractor.py        # PDF/DOCX extraction + OCR
├── preprocessing/
│   └── text_preprocessor.py     # Text cleaning pipeline
├── skill_extraction/
│   └── skill_extractor.py       # Hybrid skill extraction
├── feature_engineering/
│   └── feature_engineer.py      # BERT embeddings
├── ml_model/
│   └── classifier.py            # SVM classifier
├── matching_engine/
│   └── matcher.py               # Resume-JD matching
├── ui/
│   └── app.py                   # Streamlit web interface
├── utils/
│   ├── config_loader.py         # Configuration management
│   ├── logger.py                # Logging utilities
│   └── helpers.py               # Helper functions
├── tests/
│   ├── test_text_extraction.py
│   ├── test_preprocessing.py
│   ├── test_skill_extraction.py
│   └── test_utils.py
├── logs/                        # Application logs
├── output/                      # Results & reports
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## 🚀 Installation

### Quick Setup (TL;DR)

For experienced users, here's the complete setup in one go:

**Windows:**
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download models and data
python -m spacy download en_core_web_trf
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"

# Optional: Set environment variables
set TRANSFORMERS_NO_TF=1
set DISABLE_TORCHDYNAMO=1

# Launch application
python main.py --ui
```

**macOS/Linux:**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download models and data
python -m spacy download en_core_web_trf
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"

# Optional: Set environment variables
export TRANSFORMERS_NO_TF=1
export DISABLE_TORCHDYNAMO=1

# Launch application
python main.py --ui
```

### Detailed Installation Steps

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Tesseract OCR for scanned PDFs
- (Optional) Poppler for PDF to image conversion (required for OCR)
- **No GPU required** - runs on CPU (GPU optional for 2-3x speed)

### Step 1: Clone or Download

```bash
cd Skill_Recognition
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Set environment variables to avoid TensorFlow overhead when you only need PyTorch
> - Windows: `set TRANSFORMERS_NO_TF=1` and `set DISABLE_TORCHDYNAMO=1`
> - macOS/Linux: `export TRANSFORMERS_NO_TF=1` and `export DISABLE_TORCHDYNAMO=1`

### Step 5: Download spaCy Model

```bash
python -m spacy download en_core_web_trf
```

This downloads the transformer-based English language model required for NER.

### Step 6: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

### Step 7: (Optional but Recommended) Install Poppler

Required for OCR functionality with scanned PDFs.

**Windows:**
```bash
# 1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
# 2. Extract to C:\poppler (or your preferred location)
# 3. Update config/config.yaml with your poppler path:
#    poppler_path: "C:/poppler/Library/bin"
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### Step 8: (Optional) Install Tesseract OCR

For scanned/image-based PDFs.

**Windows:**
```bash
# 1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
# 2. Install to default location: C:\Program Files\Tesseract-OCR
# 3. Update config/config.yaml if installed elsewhere:
#    tesseract_path: "your_path/tesseract.exe"
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 9: Verify Installation

```bash
python -c "import torch; import transformers; import streamlit; import spacy; print('✓ All core packages installed successfully!')"
```

### Step 10: First Run - Model Download

On the first run, the system will automatically download pre-trained models from Hugging Face:
- `microsoft/deberta-v3-base` (~350 MB)
- `roberta-base` (~500 MB)
- `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)

**This is a one-time download** and may take 5-15 minutes depending on your internet speed. Models are cached in the `models/` directory for subsequent runs.

### Troubleshooting Common Setup Issues

**Issue: ModuleNotFoundError for spacy model**
```bash
# Solution: Download the spaCy model
python -m spacy download en_core_web_trf
```

**Issue: NLTK data not found**
```bash
# Solution: Download all required NLTK data
python -c "import nltk; nltk.download('all')"
```

**Issue: OCR not working with PDFs**
- Ensure Tesseract is installed and path is correct in `config/config.yaml`
- Ensure Poppler is installed (required for pdf2image)
- On Windows, verify paths: `tesseract_path` and `poppler_path` in config file

**Issue: "ValueError: not enough values to unpack"**
- This often indicates missing spaCy model. Run: `python -m spacy download en_core_web_trf`

**Issue: Slow first run**
- Normal behavior - transformer models are being downloaded
- Subsequent runs will be much faster (2-5 seconds per resume)

**Issue: GPU not detected (optional)**
```python
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 💻 Usage

### Pre-flight Check

Before running the application, ensure:
```bash
# 1. Verify spaCy model is installed
python -m spacy validate

# 2. Check if all required directories exist
python -c "from pathlib import Path; dirs=['data/resumes', 'models', 'logs', 'output', 'config']; [Path(d).mkdir(parents=True, exist_ok=True) for d in dirs]; print('✓ All directories ready')"

# 3. Verify configuration file
python -c "from utils.config_loader import config; print('✓ Configuration loaded successfully')"
```

### Option 1: Web Interface (Recommended)

```bash
python main.py --ui
```

Then open your browser to `http://localhost:8501`

You will be prompted with a login/register screen. Accounts are stored locally in your browser (hashed) via localStorage. Register once, then log in to access the dashboard.

**Features:**
- Upload multiple resumes
- Paste job description
- View ranked candidates
- Interactive visualizations
- Export results to CSV/JSON

### Option 2: Command Line Interface

```bash
# Basic usage
python main.py --resumes resume1.pdf resume2.pdf --jd job_description.txt

# With output file
python main.py --resumes resumes/*.pdf --jd jd.txt --output results.csv

# Using job description file
python main.py --resumes resumes/*.pdf --jd job_desc.txt --output results.json
```

### Option 3: Python API

```python
from matching_engine import ResumeJDMatcher

# Initialize matcher
matcher = ResumeJDMatcher()

# Match resumes to JD
results_df = matcher.match_resumes_to_jd(
    resume_paths=['resume1.pdf', 'resume2.pdf'],
    jd_text="We are looking for a Python developer with ML experience..."
)

# Access results
print(results_df)
```

## 🧪 Running Tests

```bash
# Run all tests
python main.py --test

# Or use pytest directly
pytest tests/ -v

# Run specific test file
pytest tests/test_skill_extraction.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## ⚙️ Configuration

### Before First Run

1. **Verify Configuration File**: Check `config/config.yaml` exists with proper paths
2. **Update Paths (if needed)**:
   - `poppler_path`: Update if Poppler installed to non-default location
   - `tesseract_path`: Update if Tesseract installed to non-default location
3. **Skills Dictionary**: Review `config/skills_dictionary.json` to add domain-specific skills

### Customization Options

Edit `config/config.yaml` to customize:

- **Text Extraction**: OCR settings, supported formats, Tesseract/Poppler paths
- **Preprocessing**: Tokenization, lemmatization options
- **Skill Extraction**: NER models, confidence thresholds
- **Feature Engineering**: Embedding models, caching
- **Matching**: Category weights, similarity metrics

### Category Weights

Default weights for skill categories:
- Technical Skills: **50%**
- Tools: **30%**
- Frameworks: **15%**
- Soft Skills: **5%**

### Skill Dictionary

Customize `config/skills_dictionary.json` to add:
- Domain-specific skills
- Company-specific tools
- Industry terminology
- Synonyms and abbreviations

## 📊 Output Format

### CSV Export
```csv
rank,resume_file,overall_score,match_percentage,technical_skills_score,...
1,john_doe.pdf,0.87,87.2%,0.92,...
2,jane_smith.pdf,0.81,81.5%,0.85,...
```

### JSON Export
```json
[
  {
    "rank": 1,
    "resume_file": "john_doe.pdf",
    "overall_score": 0.87,
    "match_percentage": "87.2%",
    "matched_skills": "Python, Machine Learning, TensorFlow, ...]",
    "missing_skills": "Kubernetes, Docker, ...]"
  }
]
```

## 🎓 Technical Details

### NLP & ML Stack

- **Text Extraction**: PyMuPDF (layout + bboxes), pdfplumber, PyPDF2, python-docx, pytesseract, pdf2image, Camelot for tables
- **NLP Pipeline**: NLTK, spaCy (`en_core_web_trf`)
- **Transformers (downloaded by setup_models.py)**: `microsoft/deberta-v3-base`, `roberta-base`, `sentence-transformers/all-MiniLM-L6-v2` (all base, not fine-tuned yet)
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Similarity Fusion**: Weighted semantic + Jaccard + fuzzy + heuristic graph; meta-learner hook present (XGBoost) but inactive until a trained checkpoint is provided
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit

### CPU vs GPU

**The system runs perfectly on CPU** - no GPU required!

- **CPU Mode** (default): 2-5 seconds per resume
- **GPU Mode** (optional): 0.5-1 second per resume
- Auto-detection: System automatically uses GPU if available, otherwise CPU

GPU provides 2-3x speed improvement but is NOT mandatory.

### Algorithms

1. **Skill Extraction**
   - Named Entity Recognition (NER) with BERT
   - Pattern matching with regex
   - Dictionary-based lookup
   - Synonym normalization

2. **Matching Algorithm**
   ```
   Final_Score = (Category_Score × 0.6) + (Embedding_Similarity × 0.4)
   
   Category_Score = Σ(CategoryMatch × Weight)
   
   Embedding_Similarity = cosine(Resume_Embedding, JD_Embedding)
   ```

3. **Feature Engineering**
   - Sentence-BERT embeddings (384 dimensions)
   - L2 normalization
   - Weighted category embeddings
   - Caching for efficiency

## 🔍 Known Limitations

1. **OCR Accuracy**: Scanned PDFs depend on image quality
2. **Language**: Currently optimized for English only
3. **Domain**: Best performance with IT/technical resumes
4. **Format**: Complex PDF layouts may affect extraction
5. **Abbreviations**: Some domain-specific acronyms may be missed

## 🚧 Future Enhancements

- [ ] Multi-language support (Spanish, French, German)
- [ ] Fine-tuned domain-specific NER models
- [ ] Experience extraction (years, seniority level)
- [ ] Education and certification parsing
- [ ] Resume ranking by multiple JDs
- [ ] API deployment (FastAPI/Flask)
- [ ] Database integration (PostgreSQL)
- [ ] Real-time processing with queues
- [ ] Advanced analytics dashboard
- [ ] A/B testing framework

## 📝 License

This project is licensed under the MIT License.

## 👥 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📧 Support

For issues, questions, or feature requests, please open an issue on GitHub.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- NLTK team for NLP tools
- Sentence-Transformers library
- Streamlit for the web framework

---

**Built with ❤️ for the data science and recruitment community**
