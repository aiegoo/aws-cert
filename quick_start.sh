#!/bin/bash
# Quick start guide for Korean OCR Application

echo "================================================"
echo "Korean OCR Application - Quick Start"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "ocr_study_app.py" ]; then
    echo "Error: Please run this script from the aws-cert directory"
    exit 1
fi

# Create directories
echo "Creating project directories..."
mkdir -p data/input data/output data/chunks logs
echo "✓ Directories created"
echo ""

# Show usage
echo "Next Steps:"
echo ""
echo "1. Install Dependencies (choose one):"
echo "   a) Automated: ./install.sh"
echo "   b) Manual: pip install -r requirements.txt"
echo ""
echo "2. Place your PDF/image files in: data/input/"
echo ""
echo "3. Run OCR processing:"
echo "   python ocr_study_app.py --input data/input/your_file.pdf"
echo ""
echo "4. Check output in: data/output/"
echo ""
echo "================================================"
echo "Example Commands:"
echo "================================================"
echo ""
echo "# Process single file:"
echo "python ocr_study_app.py --input data/input/sample.pdf"
echo ""
echo "# Process all PDFs in directory:"
echo "python ocr_study_app.py --input data/input --output data/output"
echo ""
echo "# Create text chunks:"
echo "python ocr_study_app.py --input data/input --chunk 1000"
echo ""
echo "# Process with GPU (if available):"
echo "python ocr_study_app.py --input data/input --gpu"
echo ""
echo "For more information, see: OCR_README.md"
echo "================================================"
