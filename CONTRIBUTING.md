# Contributing Guidelines

Thank you for your interest in contributing to the Resume Skill Recognition System!

## How to Contribute

### Reporting Issues

1. **Search existing issues** to avoid duplicates
2. **Use the issue template** if available
3. **Include details**:
   - OS and Python version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
   - Screenshots if applicable

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain why it would be valuable
4. Discuss potential implementation approach

### Submitting Pull Requests

#### 1. Setup Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/resume-skill-recognition.git
cd resume-skill-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black flake8 mypy
```

#### 2. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

#### 3. Make Changes

- Follow the existing code style
- Add docstrings to new functions/classes
- Update type hints
- Add unit tests for new functionality
- Update documentation if needed

#### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Check code style
black . --check
flake8 .
```

#### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add skill synonym expansion

- Add synonym mapping in skill extractor
- Update skill dictionary with common abbreviations
- Add unit tests for synonym normalization
"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding/updating tests
- `refactor:` Code refactoring
- `style:` Code style changes
- `perf:` Performance improvements
- `chore:` Maintenance tasks

#### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference related issues
- Screenshots/examples if applicable
- Checklist of changes

## Code Style Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Maximum line length: **88 characters** (Black formatter)
- Use **docstrings** for modules, classes, and functions

### Docstring Format

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    More detailed description if needed.
    Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

### Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Group related tests in classes
- Use fixtures for test data

```python
class TestSkillExtractor:
    """Test cases for SkillExtractor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = SkillExtractor()
    
    def test_extract_with_known_skills(self):
        """Test extraction with known skills in text."""
        # Test implementation
        pass
```

## Project Structure

When adding new features:

- **Core logic**: Add to appropriate module (e.g., `skill_extraction/`)
- **Tests**: Add to `tests/test_<module>.py`
- **Configuration**: Update `config/config.yaml` if needed
- **Documentation**: Update relevant `.md` files
- **Examples**: Add to `examples.py` if applicable

## Development Workflow

1. **Create issue** describing the change
2. **Discuss approach** with maintainers
3. **Implement changes** in feature branch
4. **Write tests** and ensure they pass
5. **Update documentation**
6. **Submit PR** for review
7. **Address feedback** from reviewers
8. **Merge** once approved

## Code Review Process

Pull requests will be reviewed for:

- **Functionality**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Code quality**: Is it clean and maintainable?
- **Documentation**: Is it well-documented?
- **Performance**: Are there any performance concerns?
- **Security**: Are there any security issues?

## Areas for Contribution

### High Priority

- [ ] Multi-language support
- [ ] Fine-tuned NER models
- [ ] Experience extraction
- [ ] Performance optimizations
- [ ] Additional test coverage

### Medium Priority

- [ ] Education parsing
- [ ] Certification extraction
- [ ] Database integration
- [ ] API development
- [ ] Advanced analytics

### Good First Issues

- [ ] Add more skills to dictionary
- [ ] Improve error messages
- [ ] Add more example scripts
- [ ] Fix documentation typos
- [ ] Add more unit tests

## Questions?

- Open a **discussion** on GitHub
- Join our **community chat**
- Email the maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
