"""
System Health Check Script for Resume Skill Recognition System
Verifies that all components are properly installed and configured.
"""

import sys
from pathlib import Path
from typing import List, Tuple


class HealthChecker:
    """System health checker."""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
    
    def check(self, name: str, check_func) -> bool:
        """
        Run a health check.
        
        Args:
            name: Name of the check
            check_func: Function to run
            
        Returns:
            True if passed, False otherwise
        """
        try:
            check_func()
            self.checks_passed.append(name)
            print(f"✓ {name}")
            return True
        except Exception as e:
            self.checks_failed.append((name, str(e)))
            print(f"✗ {name}: {str(e)}")
            return False
    
    def report(self):
        """Print health check report."""
        print("\n" + "="*80)
        print("HEALTH CHECK REPORT")
        print("="*80 + "\n")
        
        print(f"Passed: {len(self.checks_passed)}")
        print(f"Failed: {len(self.checks_failed)}")
        
        if self.checks_failed:
            print("\n⚠️  Failed Checks:")
            for name, error in self.checks_failed:
                print(f"  • {name}")
                print(f"    Error: {error}")
        
        print("\n" + "="*80 + "\n")
        
        if not self.checks_failed:
            print("✅ All checks passed! System is healthy.")
        else:
            print("⚠️  Some checks failed. Please review and fix issues.")
        
        return len(self.checks_failed) == 0


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        raise Exception(f"Python 3.8+ required, found {version.major}.{version.minor}")


def check_core_packages():
    """Check core packages are installed."""
    packages = [
        'numpy', 'pandas', 'scipy', 'sklearn',
        'nltk', 'transformers', 'torch', 'sentence_transformers',
        'streamlit', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        raise Exception(f"Missing packages: {', '.join(missing)}")


def check_nltk_data():
    """Check NLTK data is downloaded."""
    import nltk
    
    resources = ['punkt', 'stopwords', 'wordnet']
    missing = []
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                          else f'corpora/{resource}')
        except LookupError:
            missing.append(resource)
    
    if missing:
        raise Exception(f"Missing NLTK data: {', '.join(missing)}")


def check_project_structure():
    """Check project structure is correct."""
    required_dirs = [
        'config', 'text_extraction', 'preprocessing',
        'skill_extraction', 'feature_engineering', 'ml_model',
        'matching_engine', 'ui', 'utils', 'tests', 'data'
    ]
    
    project_root = Path(__file__).parent
    missing = []
    
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            missing.append(dir_name)
    
    if missing:
        raise Exception(f"Missing directories: {', '.join(missing)}")


def check_config_files():
    """Check configuration files exist."""
    project_root = Path(__file__).parent
    
    required_files = [
        'config/config.yaml',
        'config/skills_dictionary.json'
    ]
    
    missing = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing.append(file_path)
    
    if missing:
        raise Exception(f"Missing config files: {', '.join(missing)}")


def check_imports():
    """Check all modules can be imported."""
    modules = [
        'utils',
        'text_extraction',
        'preprocessing',
        'skill_extraction',
        'feature_engineering',
        'ml_model',
        'matching_engine'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
        except Exception as e:
            failed.append(f"{module}: {str(e)}")
    
    if failed:
        raise Exception(f"Import errors: {'; '.join(failed)}")


def check_gpu_available():
    """Check if GPU is available (optional)."""
    import torch
    
    if not torch.cuda.is_available():
        print("    Note: Running on CPU (GPU not available)")
    else:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")


def check_disk_space():
    """Check available disk space."""
    import shutil
    
    project_root = Path(__file__).parent
    stats = shutil.disk_usage(project_root)
    
    free_gb = stats.free / (1024**3)
    
    if free_gb < 5:
        raise Exception(f"Low disk space: {free_gb:.1f} GB available (5 GB minimum)")


def check_write_permissions():
    """Check write permissions."""
    project_root = Path(__file__).parent
    
    test_dirs = ['logs', 'output', 'models']
    
    for dir_name in test_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        
        test_file = dir_path / '.write_test'
        try:
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            raise Exception(f"Cannot write to {dir_name}: {str(e)}")


def main():
    """Run all health checks."""
    print("\n🏥 Resume Skill Recognition System - Health Check\n")
    print("="*80 + "\n")
    
    checker = HealthChecker()
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Core Packages", check_core_packages),
        ("NLTK Data", check_nltk_data),
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_config_files),
        ("Module Imports", check_imports),
        ("Disk Space", check_disk_space),
        ("Write Permissions", check_write_permissions),
    ]
    
    for name, check_func in checks:
        checker.check(name, check_func)
    
    # Optional checks
    print("\nOptional Components:")
    checker.check("GPU Availability", check_gpu_available)
    
    # Print report
    success = checker.report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
