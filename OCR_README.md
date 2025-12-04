# Korean OCR Study Application

OCR application for processing Korean text files (PDFs and images) with support for chunked text processing. Based on proven scripts from the Jeju Folklore project.

## Features

- 🔤 **Korean OCR**: Optimized for Korean text using EasyOCR
- 📄 **PDF Processing**: Extract text from PDF documents with high-quality rendering
- 🖼️ **Image Support**: Process PNG, JPG, TIFF, and other image formats
- 📊 **Batch Processing**: Process entire directories of files
- 🔗 **Text Chunking**: Split long texts into manageable chunks with overlap
- 📈 **Confidence Scoring**: OCR confidence scores for quality assessment
- 🚀 **GPU Acceleration**: Optional CUDA support for faster processing
- 📝 **JSON Output**: Structured JSON output with metadata

## Quick Start

### Installation

#### Automated Installation (Linux)
```bash
# Make install script executable
chmod +x install.sh

# Run installation
./install.sh
```

#### Manual Installation

1. **Install System Dependencies** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-kor \
    libtesseract-dev poppler-utils \
    libgl1-mesa-glx libglib2.0-0
```

2. **Create Virtual Environment**:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Create Project Directories**:
```bash
mkdir -p data/input data/output data/chunks logs
```

### First Run

```bash
# Activate virtual environment
source venv/bin/activate

# Test with a sample file
python ocr_study_app.py --input data/input/sample.pdf
```

## Usage

### Basic Commands

#### Process Single File
```bash
python ocr_study_app.py --input data/input/document.pdf
```

#### Process Directory
```bash
python ocr_study_app.py --input data/input --output data/output
```

#### Process Images
```bash
python ocr_study_app.py --input data/input --pattern "*.png"
```

### Advanced Options

#### Use GPU Acceleration
```bash
python ocr_study_app.py --input data/input --gpu
```

#### Create Text Chunks
```bash
# Create 1000-character chunks with 100-character overlap
python ocr_study_app.py --input data/input --chunk 1000 --chunk-overlap 100
```

#### Process Limited Files (Testing)
```bash
# Process only first 5 files
python ocr_study_app.py --input data/input --limit 5
```

#### High-Quality PDF Processing
```bash
# Use 300 DPI for better quality (slower)
python ocr_study_app.py --input data/input/document.pdf --dpi 300
```

#### Multi-Language Processing
```bash
# Process Korean and English text
python ocr_study_app.py --input data/input --languages ko,en
```

### Command-Line Options

```
--input, -i        Input file or directory path (required)
--output, -o       Output directory (default: data/output)
--pattern, -p      File pattern for directory processing (default: *.pdf)
--gpu              Use GPU acceleration (requires CUDA)
--languages, -l    Comma-separated language codes (default: ko,en)
--dpi              DPI for PDF rendering (default: 200)
--limit            Limit number of files to process
--chunk            Create text chunks of specified size
--chunk-overlap    Overlap between chunks (default: 100)
```

## Output Format

The application generates JSON files with the following structure:

### For PDF Files
```json
{
  "filename": "document.pdf",
  "total_pages": 3,
  "full_text": "Complete extracted text...",
  "pages": [
    {
      "page_number": 1,
      "text": "Page text...",
      "avg_confidence": 0.95,
      "segments": [
        {
          "text": "Text segment",
          "confidence": 0.96,
          "bbox": [[x1,y1], [x2,y2], ...]
        }
      ]
    }
  ],
  "metadata": {
    "processed_at": "2025-12-05T...",
    "dpi": 200,
    "languages": ["ko", "en"]
  }
}
```

### For Image Files
```json
{
  "filename": "image.png",
  "full_text": "Extracted text...",
  "avg_confidence": 0.94,
  "segments": [
    {
      "text": "Text segment",
      "confidence": 0.95,
      "bbox": [[x1,y1], [x2,y2], ...]
    }
  ],
  "metadata": {
    "processed_at": "2025-12-05T...",
    "languages": ["ko", "en"],
    "image_size": [1920, 1080]
  }
}
```

### With Chunking Enabled
```json
{
  "source": "document.json",
  "chunks": [
    {
      "chunk_id": 0,
      "start_pos": 0,
      "end_pos": 1000,
      "text": "Chunk text...",
      "char_count": 1000
    }
  ]
}
```

## Project Structure

```
aws-cert/
├── ocr_study_app.py      # Main application
├── requirements.txt       # Python dependencies
├── install.sh            # Installation script
├── OCR_README.md         # This file
├── data/
│   ├── input/           # Place input files here
│   ├── output/          # Processed JSON files
│   └── chunks/          # Chunked text files
├── logs/
│   └── ocr_app.log      # Application logs
└── venv/                # Virtual environment (created during install)
```

## Dependencies

### Core OCR
- EasyOCR 1.7.0+ - Korean/English OCR engine
- PyMuPDF 1.26.0+ - PDF processing
- Tesseract OCR - System-level OCR support

### Image Processing
- OpenCV Python 4.8.0+
- Pillow 10.0.0+
- NumPy 1.24.0+

### Korean Text Processing
- KoNLPy 0.6.0+ - Korean NLP toolkit

See `requirements.txt` for complete list.

## Performance Tips

1. **GPU Acceleration**: Use `--gpu` flag if you have CUDA-capable GPU (10-50x faster)
2. **DPI Settings**: Lower DPI (150-200) for faster processing, higher (300+) for better accuracy
3. **Batch Processing**: Process multiple files at once for better throughput
4. **Chunking**: Use chunking for very large documents to manage memory

## Troubleshooting

### Installation Issues

**Issue**: `EasyOCR model download fails`
```bash
# Pre-download models manually
python3 -c "import easyocr; easyocr.Reader(['ko', 'en'], download_enabled=True)"
```

**Issue**: `Tesseract not found`
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

**Issue**: `OpenCV import error`
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### Runtime Issues

**Issue**: Low OCR accuracy
- Try increasing DPI: `--dpi 300`
- Ensure input images are clear and high-resolution
- Check language settings: `--languages ko,en`

**Issue**: Out of memory
- Process files one at a time instead of batch
- Use chunking: `--chunk 1000`
- Disable GPU: remove `--gpu` flag

**Issue**: Slow processing
- Enable GPU: `--gpu` (requires CUDA)
- Lower DPI: `--dpi 150`
- Process smaller batches: `--limit 10`

## Related Projects

This application is based on scripts from:
- `ocr-proven/team-data/jeju-stories` - Jeju Folklore OCR pipeline
- Proven in production for processing 1,555+ Korean PDF documents

## License

This project is part of the AWS Certification study materials.

## Support

For issues or questions:
1. Check the logs in `logs/ocr_app.log`
2. Review troubleshooting section above
3. Verify installation: `python ocr_study_app.py --help`

## Examples

### Example 1: Process AWS Study Materials
```bash
# Process all PDFs in general/ directory
python ocr_study_app.py \
    --input general/ \
    --output processed_materials/ \
    --chunk 2000
```

### Example 2: Extract Text from MLA Materials
```bash
# Process ML Engineering materials with high quality
python ocr_study_app.py \
    --input mla/ \
    --dpi 300 \
    --gpu
```

### Example 3: Quick Test Run
```bash
# Test with first 3 files
python ocr_study_app.py \
    --input devops/ \
    --limit 3 \
    --output test_output/
```

---

**Happy OCR Processing! 🚀**
