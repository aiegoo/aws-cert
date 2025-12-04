#!/bin/bash
# Korean OCR Study Application - Installation Script
# For Ubuntu/Debian-based systems

set -e  # Exit on error

echo "================================================"
echo "Korean OCR Study Application - Installation"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_warning "This script is optimized for Linux. Some steps may need adjustment."
fi

# 1. System package dependencies
echo ""
echo "Step 1: Installing system dependencies..."
print_status "Updating package list..."
sudo apt-get update

print_status "Installing system packages..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-kor \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

print_status "System dependencies installed"

# 2. Create and activate virtual environment
echo ""
echo "Step 2: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

source venv/bin/activate
print_status "Virtual environment activated"

# 3. Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_status "Pip upgraded"

# 4. Install Python dependencies
echo ""
echo "Step 4: Installing Python packages..."
print_status "This may take several minutes (downloading OCR models)..."
pip install -r requirements.txt
print_status "Python packages installed"

# 5. Download EasyOCR models
echo ""
echo "Step 5: Pre-downloading EasyOCR Korean models..."
python3 << EOF
import easyocr
print("Initializing EasyOCR with Korean and English models...")
reader = easyocr.Reader(['ko', 'en'], gpu=False, download_enabled=True)
print("Models downloaded successfully!")
EOF
print_status "OCR models ready"

# 6. Create necessary directories
echo ""
echo "Step 6: Creating project directories..."
mkdir -p data/input data/output data/chunks logs
print_status "Directories created"

# 7. Set up configuration
echo ""
echo "Step 7: Creating configuration file..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Korean OCR Application Configuration
DEBUG=False
LOG_LEVEL=INFO
USE_GPU=False
OUTPUT_FORMAT=json
CHUNK_SIZE=1000
OCR_LANGUAGE=ko,en
EOF
    print_status "Configuration file created (.env)"
else
    print_warning "Configuration file already exists"
fi

# 8. Verify installation
echo ""
echo "Step 8: Verifying installation..."
python3 << EOF
import sys
try:
    import fitz
    import easyocr
    import PIL
    import cv2
    import numpy as np
    print("✓ All core dependencies imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_status "Installation verification passed"
else
    print_error "Installation verification failed"
    exit 1
fi

# 9. Installation complete
echo ""
echo "================================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To test the OCR application, run:"
echo "    python ocr_study_app.py --help"
echo ""
echo "To deactivate the environment, run:"
echo "    deactivate"
echo ""
print_status "Installation successful!"
