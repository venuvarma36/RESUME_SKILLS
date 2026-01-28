# Quick Start Guide

This guide will help you get started with the Resume Skill Recognition System in under 5 minutes.

## 1. Installation (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 2. Launch the Web UI (1 minute)

```bash
python main.py --ui
```

Open browser to: `http://localhost:8501`

## 3. Try the Example (2 minutes)

### Option A: Using the Web Interface

1. Click "Browse files" in the sidebar
2. Upload one or more resume files (PDF/DOCX)
3. Paste a job description in the text area
4. Click "Match Resumes"
5. View results and export

### Option B: Using Python Code

```python
from matching_engine import ResumeJDMatcher

# Initialize
matcher = ResumeJDMatcher()

# Job description
jd_text = """
We need a Python developer with Machine Learning experience.
Required skills: Python, TensorFlow, Django, Docker, AWS.
"""

# Match resumes
results = matcher.match_resumes_to_jd(
    resume_paths=['resume1.pdf', 'resume2.pdf'],
    jd_text=jd_text
)

# View results
print(results[['rank', 'resume_file', 'match_percentage']])
```

## 4. Run Examples

```bash
python examples.py
```

This will demonstrate:
- Basic skill extraction
- Resume file processing
- Text preprocessing
- Resume-JD matching
- Batch processing

## What's Next?

- **Customize Skills**: Edit `config/skills_dictionary.json` to add domain-specific skills
- **Adjust Weights**: Modify `config/config.yaml` to change category weights
- **Add Resumes**: Place your resume files in `data/resumes/`
- **Run Tests**: Execute `python main.py --test` to verify installation

## Common Issues

### Issue: "Module not found"
**Solution**: Make sure you installed all requirements: `pip install -r requirements.txt`

### Issue: "NLTK data not found"
**Solution**: Download NLTK data:
```python
import nltk
nltk.download('all')
```

### Issue: "OCR not working"
**Solution**: Install Tesseract OCR:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

### Issue: "Model loading slow"
**Solution**: First run downloads transformer models (1-2 GB). Subsequent runs will be faster.

**Note**: This is normal and happens once. The system runs on CPU by default (no GPU required).

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Run `python main.py --help` for CLI options
- Open an issue on GitHub for bugs or questions

## Pro Tips

1. **No GPU needed**: System runs perfectly on CPU. GPU only provides 2-3x speed boost (optional)
2. **Performance**: Use GPU for faster processing (CUDA-enabled GPU + PyTorch with CUDA)
3. **Accuracy**: Add company-specific skills to the dictionary
4. **Batch Processing**: Process multiple resumes at once for better efficiency
5. **Export**: Always export results to CSV for further analysis

Happy Matching! 🚀
