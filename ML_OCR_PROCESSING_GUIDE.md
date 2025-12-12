# ML Engineering on AWS - OCR Processing Guide

## Overview

This guide explains how to process the 520-page ML Engineering on AWS PDF using OCR to create interactive study materials for MLA-C01 exam preparation.

## 📊 Statistics

- **Total Pages**: 520
- **Total Chunks**: 26 (20 pages each)
- **Processing Time**: ~3-5 hours on CPU (20-40 seconds per page)
- **Output Format**: JSON + Interactive HTML

## 🔧 Prerequisites

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Verify installations
python3 -c "import easyocr; print('EasyOCR:', easyocr.__version__)"
python3 -c "import fitz; print('PyMuPDF:', fitz.version)"
```

## 📁 Directory Structure

```
aws-cert/
├── chunks_ML Engineering on AWS/        # Input PDF chunks (26 files)
│   ├── ML Engineering on AWS_chunk_001_pages_1-20.pdf
│   ├── ML Engineering on AWS_chunk_002_pages_21-40.pdf
│   └── ... (24 more files)
│
├── processed_materials/                  # OCR outputs
│   └── ml_engineering_full/
│       ├── chunk_001/                   # Individual chunk outputs
│       │   ├── chunk_001.json
│       │   └── chunk_001.txt
│       ├── chunk_002/
│       └── ...
│       ├── processing_progress.json      # Resume tracking
│       └── ml_engineering_full_ocr.json  # Consolidated output
│
└── ml_study_materials/                   # Generated study materials
    ├── ml_engineering_reader.html        # Interactive reader
    ├── ml_engineering_flashcards.html    # Flashcards (future)
    └── ml_engineering_quiz.html          # Quiz (future)
```

## 🚀 Processing Workflow

### Step 1: Batch OCR Processing

Process all 26 PDF chunks with automatic resume capability:

```bash
# Process all chunks (can resume if interrupted)
python3 batch_ocr_processor.py \
  --chunks-dir "chunks_ML Engineering on AWS" \
  --output processed_materials/ml_engineering_full \
  --dpi 150

# Optional: Start fresh (ignore previous progress)
python3 batch_ocr_processor.py --no-resume

# Optional: Only consolidate existing outputs
python3 batch_ocr_processor.py --consolidate-only
```

**Features:**
- ✅ Automatic progress tracking (resumes from last completed chunk)
- ✅ Error handling and retry capability
- ✅ Consolidated JSON output
- ✅ Detailed logging (`batch_ocr_processing.log`)

### Step 2: Generate Study Materials

Create interactive HTML study materials from OCR output:

```bash
# Generate interactive reader and flashcards
python3 generate_study_materials.py \
  --input processed_materials/ml_engineering_full/ml_engineering_full_ocr.json \
  --output ml_study_materials
```

**Generated Materials:**
- 📖 Interactive reader with navigation and search
- 🎴 Flashcards (planned)
- 📝 Practice quizzes (planned)

## 💡 Usage Examples

### Process Specific Chunks

If you want to process specific chunks manually:

```bash
# Process chunk 1
python3 ocr_study_app.py \
  --input "chunks_ML Engineering on AWS/ML Engineering on AWS_chunk_001_pages_1-20.pdf" \
  --output processed_materials/ml_engineering_full/chunk_001 \
  --dpi 150

# Process chunk 5
python3 ocr_study_app.py \
  --input "chunks_ML Engineering on AWS/ML Engineering on AWS_chunk_005_pages_81-100.pdf" \
  --output processed_materials/ml_engineering_full/chunk_005 \
  --dpi 150
```

### Monitor Progress

```bash
# Watch processing log in real-time
tail -f batch_ocr_processing.log

# Check progress JSON
cat processed_materials/ml_engineering_full/processing_progress.json
```

### Resume After Interruption

The batch processor automatically tracks progress. If interrupted:

```bash
# Simply run again - it will resume from last completed chunk
python3 batch_ocr_processor.py
```

## 📖 Output Formats

### Individual Chunk Output

Each chunk produces:

```json
{
  "filename": "ML Engineering on AWS_chunk_001_pages_1-20.pdf",
  "total_pages": 20,
  "processing_date": "2025-12-12T10:27:00",
  "pages": [
    {
      "page_number": 1,
      "text": "Extracted text content...",
      "chunks": ["chunk1", "chunk2", ...]
    }
  ]
}
```

### Consolidated Output

All chunks combined:

```json
{
  "title": "ML Engineering on AWS",
  "total_pages": 520,
  "total_chunks": 26,
  "processed_date": "2025-12-12T15:30:00",
  "chunks": [
    {
      "chunk_number": 1,
      "chunk_dir": "chunk_001",
      "data": { ... }
    }
  ]
}
```

## 🎯 Study Materials

### Interactive Reader

The generated `ml_engineering_reader.html` includes:

- **Navigation**: Browse all sections with keyboard shortcuts (← →)
- **Search**: Find content across all 520 pages
- **Progress Tracking**: Visual progress bar
- **Table of Contents**: Quick jump to any section
- **Responsive Design**: Works on mobile and desktop

**Usage:**
```bash
# Open in browser
open ml_study_materials/ml_engineering_reader.html

# Or deploy to GitHub Pages
cp ml_study_materials/*.html .
git add .
git commit -m "Add ML Engineering study materials"
git push origin gh-pages
```

## ⚙️ Configuration Options

### OCR Quality vs Speed

```bash
# Higher quality (slower)
python3 batch_ocr_processor.py --dpi 200

# Standard quality (recommended)
python3 batch_ocr_processor.py --dpi 150

# Faster processing (lower quality)
python3 batch_ocr_processor.py --dpi 100
```

### GPU Acceleration

If you have a CUDA-enabled GPU:

```bash
# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OCR will automatically use GPU (10-20x faster)
python3 batch_ocr_processor.py
```

## 🐛 Troubleshooting

### Issue: "Out of Memory"

```bash
# Reduce DPI
python3 batch_ocr_processor.py --dpi 100

# Process fewer pages at a time
# Edit ocr_study_app.py to add memory cleanup
```

### Issue: "Timeout on Large Pages"

The batch processor has a 1-hour timeout per chunk. If needed:

```python
# Edit batch_ocr_processor.py, line 96
timeout=3600  # Increase to 7200 for 2 hours
```

### Issue: "Korean Characters Not Recognized"

```bash
# Verify EasyOCR models downloaded
ls -la ~/.EasyOCR/model/

# Should see: craft_mlt_25k.pth, korean_g2.pth, latin_g2.pth
```

## 📈 Performance Optimization

### Expected Processing Times

| Hardware | Pages/Hour | Total Time (520 pages) |
|----------|-----------|------------------------|
| CPU Only | ~90-180   | 3-6 hours              |
| GPU (CUDA) | ~1500-3000 | 10-20 minutes         |

### Batch Processing Strategy

```bash
# Run overnight for CPU processing
nohup python3 batch_ocr_processor.py > ocr_output.log 2>&1 &

# Check progress
tail -f ocr_output.log

# Or use screen/tmux for long sessions
screen -S ocr_processing
python3 batch_ocr_processor.py
# Ctrl+A, D to detach
```

## 📚 Next Steps

After OCR processing completes:

1. **Review Output Quality**
   ```bash
   # Check sample chunk
   cat processed_materials/ml_engineering_full/chunk_001/chunk_001.txt | head -100
   ```

2. **Generate Study Materials**
   ```bash
   python3 generate_study_materials.py
   ```

3. **Deploy to GitHub Pages**
   ```bash
   git checkout gh-pages
   cp ml_study_materials/ml_engineering_reader.html .
   git add ml_engineering_reader.html
   git commit -m "Add ML Engineering interactive reader"
   git push origin gh-pages
   ```

4. **Update Index Page**
   - Add link to ML Engineering reader
   - Include navigation to other study materials

## 🔗 Related Resources

- [MLA-C01 Flashcards](mla_study_cards.html) - 65 comprehensive flashcards
- [Storage Flashcards](storage_study_cards.html) - 22 lakehouse architecture cards
- [Storage Quiz](storage_quiz.html) - 20 practice questions
- [DevOps Flashcards](study_cards.html) - 30 DevOps fundamentals

## 📝 Notes

- Processing can be interrupted at any time - progress is saved
- OCR quality is good for English and Korean text
- Some complex diagrams may not be captured well
- Consider manual review of critical sections
- The batch processor is idempotent - safe to run multiple times

## 🎓 Exam Preparation Strategy

1. **Read Through** - Use interactive reader for comprehensive review
2. **Practice Flashcards** - Focus on weak areas identified during reading
3. **Take Quizzes** - Test knowledge on specific domains
4. **Review Mistakes** - Return to reader for detailed explanations
5. **Repeat** - Continue cycle until confident

---

**Happy Studying! 🚀**
