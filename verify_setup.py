#!/usr/bin/env python3
"""
Verify OCR Application Setup
Quick check to ensure all dependencies are available
"""

import sys

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    print("✓ Python version OK")
    return True

def check_imports():
    """Check if all required packages can be imported"""
    packages = {
        'fitz': 'PyMuPDF',
        'easyocr': 'EasyOCR',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'NumPy',
    }
    
    print("\nChecking Python packages:")
    all_ok = True
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_files():
    """Check if required files exist"""
    from pathlib import Path
    
    print("\nChecking project files:")
    files = {
        'ocr_study_app.py': 'Main application',
        'requirements.txt': 'Dependencies list',
        'install.sh': 'Installation script',
        'OCR_README.md': 'Documentation'
    }
    
    all_ok = True
    for file, desc in files.items():
        if Path(file).exists():
            print(f"✓ {file} - {desc}")
        else:
            print(f"✗ {file} - MISSING")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check if required directories exist"""
    from pathlib import Path
    
    print("\nChecking directories:")
    dirs = ['data/input', 'data/output', 'data/chunks', 'logs']
    
    all_ok = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ - Creating...")
            path.mkdir(parents=True, exist_ok=True)
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("Korean OCR Application - Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_imports),
        ("Project Files", check_files),
        ("Directories", check_directories),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            results.append(check_func())
        except Exception as e:
            print(f"✗ Error checking {name}: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    if all(results):
        print("✓ Setup verification PASSED")
        print("\nYou're ready to use the OCR application!")
        print("Try: python ocr_study_app.py --help")
    else:
        print("⚠️  Setup verification FAILED")
        print("\nMissing dependencies. Please run:")
        print("  ./install.sh")
        print("or")
        print("  pip install -r requirements.txt")
    print("="*60)
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    sys.exit(main())
