#!/usr/bin/env python3
"""
Korean OCR Study Application
Processes Korean text files (PDFs, images) split into chunks using OCR

Based on proven scripts from ocr-proven/team-data/jeju-stories
Optimized for Korean language processing with EasyOCR
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys

# PDF and Image Processing
try:
    import fitz  # PyMuPDF
    import easyocr
    from PIL import Image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not installed. Run: pip install -r requirements.txt")
    print(f"Missing: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ocr_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KoreanOCRProcessor:
    """Korean OCR processor for study materials"""
    
    def __init__(self, use_gpu: bool = False, languages: List[str] = ['ko', 'en']):
        """
        Initialize OCR processor
        
        Args:
            use_gpu: Use GPU acceleration if available
            languages: List of languages for OCR (default: Korean and English)
        """
        logger.info(f"Initializing EasyOCR with languages: {languages}")
        logger.info("This may take a few minutes on first run (downloading models)...")
        
        try:
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
        
        self.use_gpu = use_gpu
        self.languages = languages
        
    def extract_text_from_pdf(self, pdf_path: Path, dpi: int = 200) -> Dict[str, any]:
        """
        Extract text from PDF using OCR
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for rendering (higher = better quality, slower)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            doc = fitz.open(str(pdf_path))
            results = {
                'filename': pdf_path.name,
                'total_pages': len(doc),
                'pages': [],
                'full_text': '',
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'dpi': dpi,
                    'languages': self.languages
                }
            }
            
            full_text_parts = []
            
            for page_num in range(len(doc)):
                logger.info(f"  Processing page {page_num + 1}/{len(doc)}")
                page = doc[page_num]
                
                # Render page as image
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")
                
                # Perform OCR
                ocr_results = self.reader.readtext(img_bytes, detail=1, paragraph=True)
                
                # Extract text and confidence scores
                page_text = []
                for result in ocr_results:
                    if len(result) == 3:
                        bbox, text, conf = result
                    elif len(result) == 2:
                        bbox, text = result
                        conf = 1.0  # Default confidence if not provided
                    else:
                        continue
                    
                    page_text.append({
                        'text': text,
                        'confidence': float(conf),
                        'bbox': bbox
                    })
                
                # Combine text
                page_full_text = '\n'.join([item['text'] for item in page_text])
                full_text_parts.append(page_full_text)
                
                results['pages'].append({
                    'page_number': page_num + 1,
                    'text': page_full_text,
                    'segments': page_text,
                    'avg_confidence': np.mean([item['confidence'] for item in page_text]) if page_text else 0.0
                })
            
            doc.close()
            results['full_text'] = '\n\n'.join(full_text_parts)
            
            logger.info(f"  Extracted {len(results['full_text'])} characters")
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def extract_text_from_image(self, image_path: Path) -> Dict[str, any]:
        """
        Extract text from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info(f"Processing image: {image_path.name}")
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Perform OCR
            ocr_results = self.reader.readtext(image, detail=1, paragraph=True)
            
            # Extract text and confidence scores
            text_segments = []
            for result in ocr_results:
                if len(result) == 3:
                    bbox, text, conf = result
                elif len(result) == 2:
                    bbox, text = result
                    conf = 1.0  # Default confidence if not provided
                else:
                    continue
                
                text_segments.append({
                    'text': text,
                    'confidence': float(conf),
                    'bbox': bbox
                })
            
            full_text = '\n'.join([seg['text'] for seg in text_segments])
            avg_confidence = np.mean([seg['confidence'] for seg in text_segments]) if text_segments else 0.0
            
            results = {
                'filename': image_path.name,
                'full_text': full_text,
                'segments': text_segments,
                'avg_confidence': float(avg_confidence),
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'languages': self.languages,
                    'image_size': image.shape[:2]
                }
            }
            
            logger.info(f"  Extracted {len(full_text)} characters (confidence: {avg_confidence:.2%})")
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         file_pattern: str = "*.pdf", limit: Optional[int] = None) -> List[Path]:
        """
        Process all files in directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match (e.g., "*.pdf", "*.png")
            limit: Maximum number of files to process
            
        Returns:
            List of output file paths
        """
        logger.info(f"Processing directory: {input_dir}")
        logger.info(f"  Pattern: {file_pattern}")
        logger.info(f"  Output: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        files = sorted(input_dir.glob(file_pattern))
        if limit:
            files = files[:limit]
            logger.info(f"  Limiting to {limit} files")
        
        logger.info(f"  Found {len(files)} files to process")
        
        output_files = []
        
        for idx, file_path in enumerate(files, 1):
            logger.info(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")
            
            try:
                # Determine file type and process
                if file_path.suffix.lower() == '.pdf':
                    results = self.extract_text_from_pdf(file_path)
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    results = self.extract_text_from_image(file_path)
                else:
                    logger.warning(f"  Skipping unsupported file type: {file_path.suffix}")
                    continue
                
                # Save results
                output_file = output_dir / f"{file_path.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                output_files.append(output_file)
                logger.info(f"  ✓ Saved: {output_file.name}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete!")
        logger.info(f"  Processed: {len(output_files)}/{len(files)} files")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"{'='*60}")
        
        return output_files
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'chunk_id': chunk_id,
                'start_pos': start,
                'end_pos': end,
                'text': chunk_text,
                'char_count': len(chunk_text)
            })
            
            chunk_id += 1
            start += chunk_size - overlap
        
        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Korean OCR Study Application - Process Korean text files with OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF file
  python ocr_study_app.py --input data/input/document.pdf
  
  # Process all PDFs in directory
  python ocr_study_app.py --input data/input --pattern "*.pdf"
  
  # Process with GPU acceleration
  python ocr_study_app.py --input data/input --gpu
  
  # Limit to first 5 files for testing
  python ocr_study_app.py --input data/input --limit 5
  
  # Process and create text chunks
  python ocr_study_app.py --input data/input --chunk 1000
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', '-o', type=str, default='data/output',
                       help='Output directory (default: data/output)')
    parser.add_argument('--pattern', '-p', type=str, default='*.pdf',
                       help='File pattern for directory processing (default: *.pdf)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration (requires CUDA)')
    parser.add_argument('--languages', '-l', type=str, default='ko,en',
                       help='Comma-separated language codes (default: ko,en)')
    parser.add_argument('--dpi', type=int, default=200,
                       help='DPI for PDF rendering (default: 200)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of files to process')
    parser.add_argument('--chunk', type=int,
                       help='Create text chunks of specified size')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='Overlap between chunks (default: 100)')
    
    args = parser.parse_args()
    
    # Parse input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    output_path = Path(args.output)
    languages = [lang.strip() for lang in args.languages.split(',')]
    
    # Initialize processor
    try:
        processor = KoreanOCRProcessor(use_gpu=args.gpu, languages=languages)
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {e}")
        return 1
    
    # Process input
    try:
        if input_path.is_file():
            # Process single file
            logger.info("Processing single file...")
            output_path.mkdir(parents=True, exist_ok=True)
            
            if input_path.suffix.lower() == '.pdf':
                results = processor.extract_text_from_pdf(input_path, dpi=args.dpi)
            else:
                results = processor.extract_text_from_image(input_path)
            
            # Apply chunking if requested
            if args.chunk:
                chunks = processor.chunk_text(results['full_text'], 
                                            chunk_size=args.chunk,
                                            overlap=args.chunk_overlap)
                results['chunks'] = chunks
            
            # Save results
            output_file = output_path / f"{input_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Results saved to: {output_file}")
            
        elif input_path.is_dir():
            # Process directory
            logger.info("Processing directory...")
            output_files = processor.process_directory(
                input_path, output_path, 
                file_pattern=args.pattern,
                limit=args.limit
            )
            
            # Apply chunking if requested
            if args.chunk:
                logger.info(f"\nCreating chunks (size: {args.chunk}, overlap: {args.chunk_overlap})...")
                chunks_dir = output_path / 'chunks'
                chunks_dir.mkdir(exist_ok=True)
                
                for output_file in output_files:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    chunks = processor.chunk_text(data['full_text'],
                                                chunk_size=args.chunk,
                                                overlap=args.chunk_overlap)
                    
                    chunk_file = chunks_dir / f"{output_file.stem}_chunks.json"
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json.dump({'source': output_file.name, 'chunks': chunks}, 
                                f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"  ✓ Chunks saved: {chunk_file.name}")
        
        logger.info("\n✓ All processing complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
