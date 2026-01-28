# Installation Guide

Detailed instructions for setting up the Resume Skill Recognition System.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Dependency Installation](#dependency-installation)
4. [NLTK Data Setup](#nltk-data-setup)
5. [Optional: OCR Setup](#optional-ocr-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements (CPU-Only)
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Disk Space**: 5 GB free space
- **Internet**: Required for initial model downloads
- **GPU**: Not required - runs perfectly on CPU

### Recommended Requirements (for faster processing)
- **Python**: 3.10+
- **RAM**: 16 GB or more
- **GPU**: CUDA-compatible GPU (optional, 2-3x speed improvement)
- **Disk Space**: 10 GB free space

> **Note**: The system automatically detects GPU availability and uses CPU as fallback. GPU is completely optional!

## Python Environment Setup

### Option 1: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```


## Dependency Installation

### Step 1: Core Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core packages
pip install -r requirements.txt
```

This will install:
- NumPy, Pandas, SciPy (numerical computing)
- NLTK (natural language processing)
- Transformers, Torch (deep learning)
- Scikit-learn (machine learning)
- Streamlit (web interface)
- And more...

### Step 2: Verify Installation

```python
python -c "import torch; import transformers; import streamlit; print('All packages installed successfully!')"
```

## NLTK Data Setup

NLTK requires additional data files for tokenization, stopwords, etc.

### Automated Download

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

### Manual Download (if automated fails)

```python
import nltk
nltk.download()  # Opens GUI to select packages
```

Select and download:
- punkt
- stopwords
- wordnet
- averaged_perceptron_tagger
- omw-1.4

## Optional: OCR Setup

For processing scanned PDFs, install Tesseract OCR.

### Windows

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (choose default options)
3. Add to PATH: `C:\Program Files\Tesseract-OCR`
4. Install Python wrapper:
   ```bash
   pip install pdf2image
   ```
5. Install Poppler (required for pdf2image):
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract and add `bin` folder to PATH

### macOS

```bash
# Install Tesseract
brew install tesseract

# Install Poppler
brew install poppler

# Install Python wrapper
pip install pdf2image
```

### Linux (Ubuntu/Debian)

```bash
# Install Tesseract
sudo apt-get update
sudo apt-get install tesseract-ocr

# Install Poppler
sudo apt-get install poppler-utils

# Install Python wrapper
pip install pdf2image
```

### Verify OCR Installation

```python
python -c "import pytesseract; from pdf2image import convert_from_path; print('OCR setup successful!')"
```

## Verification

### Run Quick Test

```bash
# Test basic functionality
python -c "from matching_engine import ResumeJDMatcher; print('System ready!')"

# Run example scripts
python examples.py

# Run unit tests
python -m pytest tests/ -v
```

### Expected Output

You should see:
- No import errors
- Example outputs showing skill extraction
- Test results (all passing)

## Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: NLTK Resource Not Found

**Problem**: NLTK data not downloaded

**Solution**:
```python
import nltk
nltk.download('all')  # Download all data (may take time)
```

### Issue: Torch/Transformers Import Error

**Problem**: PyTorch or Transformers not properly installed

**Solution**:
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch

# Reinstall Transformers
pip uninstall transformers
pip install transformers
```

### Issue: Slow Model Loading

**Problem**: First-time model downloads are slow

**Solution**: Be patient. Models are 1-2 GB and need to be downloaded once.
Subsequent runs will be much faster.

### Issue: CUDA Not Available

**Problem**: GPU not detected (if you have a CUDA GPU)

**Solution**:
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Streamlit Port Already in Use

**Problem**: Port 8501 is occupied

**Solution**:
```bash
# Use different port
streamlit run ui/app.py --server.port 8502
```

### Issue: Permission Denied

**Problem**: Cannot create directories or files

**Solution**:
- Windows: Run as Administrator
- macOS/Linux: Use `sudo` or check directory permissions

## GPU Acceleration (Optional - For Speed Only)

**Important**: GPU is NOT required. The system runs perfectly fine on CPU!

GPU only provides 2-3x speed improvement for large-scale processing. For typical use cases (processing 10-20 resumes), CPU performance is adequate.

### Check Current Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
```

### If You Want GPU Acceleration (Optional)

Only follow these steps if you have an NVIDIA GPU and need faster processing:

1. Download from: https://developer.nvidia.com/cuda-downloads
2. Install CUDA Toolkit (version 11.8 recommended)
3. Install cuDNN
4. Reinstall PyTorch with CUDA support

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for quick start guide
2. Check [README.md](README.md) for detailed documentation
3. Run `python examples.py` to see usage examples
4. Start the web UI: `python main.py --ui`

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

## Additional Resources

- [Python Documentation](https://docs.python.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
