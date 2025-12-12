#!/usr/bin/env python3
"""
Batch OCR Processor for ML Engineering on AWS PDF chunks
Processes all 26 chunks and generates consolidated study materials
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ocr_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BatchOCRProcessor:
    """Batch process PDF chunks with OCR and generate study materials"""
    
    def __init__(self, chunks_dir: str, output_base: str):
        self.chunks_dir = Path(chunks_dir)
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_base / 'processing_progress.json'
        self.progress = self.load_progress()
        
    def load_progress(self) -> Dict:
        """Load processing progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'completed_chunks': [],
            'failed_chunks': [],
            'last_updated': None
        }
    
    def save_progress(self):
        """Save processing progress to file"""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_chunk_files(self) -> List[Path]:
        """Get all PDF chunk files sorted by chunk number"""
        chunks = sorted(self.chunks_dir.glob('ML Engineering on AWS_chunk_*.pdf'))
        logger.info(f"Found {len(chunks)} PDF chunks to process")
        return chunks
    
    def is_chunk_processed(self, chunk_name: str) -> bool:
        """Check if chunk has already been processed"""
        return chunk_name in self.progress['completed_chunks']
    
    def process_chunk(self, chunk_file: Path, dpi: int = 150) -> bool:
        """Process a single PDF chunk with OCR"""
        chunk_name = chunk_file.stem
        
        if self.is_chunk_processed(chunk_name):
            logger.info(f"Skipping already processed: {chunk_name}")
            return True
        
        # Extract chunk number for output directory
        chunk_num = chunk_name.split('_chunk_')[1].split('_')[0]
        output_dir = self.output_base / f"chunk_{chunk_num}"
        
        logger.info(f"Processing {chunk_name}...")
        logger.info(f"Output directory: {output_dir}")
        
        # Build OCR command
        cmd = [
            'python3', 'ocr_study_app.py',
            '--input', str(chunk_file),
            '--output', str(output_dir),
            '--dpi', str(dpi)
        ]
        
        try:
            # Run OCR process with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"  {line}")
            
            # Wait for completion
            return_code = process.wait(timeout=3600)
            
            if return_code == 0:
                logger.info(f"✓ Successfully processed {chunk_name}")
                self.progress['completed_chunks'].append(chunk_name)
                self.save_progress()
                return True
            else:
                logger.error(f"✗ Failed to process {chunk_name} (exit code: {return_code})")
                self.progress['failed_chunks'].append(chunk_name)
                self.save_progress()
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout processing {chunk_name}")
            if process:
                process.kill()
            self.progress['failed_chunks'].append(chunk_name)
            self.save_progress()
            return False
        except Exception as e:
            logger.error(f"✗ Error processing {chunk_name}: {e}")
            self.progress['failed_chunks'].append(chunk_name)
            self.save_progress()
            return False
    
    def process_all_chunks(self, dpi: int = 150, resume: bool = True):
        """Process all chunks in sequence"""
        chunks = self.get_chunk_files()
        total = len(chunks)
        
        logger.info(f"Starting batch OCR processing of {total} chunks")
        logger.info(f"Resume mode: {resume}")
        
        if resume and self.progress['completed_chunks']:
            logger.info(f"Resuming from previous run - {len(self.progress['completed_chunks'])} chunks already completed")
        
        success_count = len(self.progress['completed_chunks'])
        fail_count = 0
        
        for i, chunk_file in enumerate(chunks, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Chunk {i}/{total}: {chunk_file.name}")
            logger.info(f"{'='*60}")
            
            success = self.process_chunk(chunk_file, dpi)
            
            if success and chunk_file.stem not in self.progress['completed_chunks']:
                success_count += 1
            elif not success:
                fail_count += 1
            
            # Progress summary
            remaining = total - i
            logger.info(f"\nProgress: {i}/{total} chunks processed")
            logger.info(f"Success: {success_count}, Failed: {fail_count}, Remaining: {remaining}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Batch processing complete!")
        logger.info(f"Total success: {success_count}/{total}")
        logger.info(f"Total failed: {fail_count}/{total}")
        logger.info(f"{'='*60}")
        
        return success_count, fail_count
    
    def consolidate_output(self) -> Dict:
        """Consolidate all chunk outputs into a single JSON"""
        logger.info("Consolidating OCR output from all chunks...")
        
        consolidated = {
            'title': 'ML Engineering on AWS',
            'total_pages': 520,
            'total_chunks': 26,
            'processed_date': datetime.now().isoformat(),
            'chunks': []
        }
        
        # Collect all chunk outputs
        for chunk_dir in sorted(self.output_base.glob('chunk_*')):
            chunk_num = chunk_dir.name.split('_')[1]
            
            # Find the JSON file in the chunk directory
            json_files = list(chunk_dir.glob('*.json'))
            
            if json_files:
                json_file = json_files[0]  # Take the first JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    consolidated['chunks'].append({
                        'chunk_number': int(chunk_num),
                        'chunk_dir': chunk_dir.name,
                        'data': chunk_data
                    })
                logger.info(f"✓ Added chunk {chunk_num}")
            else:
                logger.warning(f"✗ Missing output for chunk {chunk_num}")
        
        # Save consolidated output
        output_file = self.output_base / 'ml_engineering_full_ocr.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Consolidated output saved to: {output_file}")
        logger.info(f"Total chunks consolidated: {len(consolidated['chunks'])}")
        
        return consolidated


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch OCR processor for ML Engineering PDF chunks')
    parser.add_argument('--chunks-dir', default='chunks_ML Engineering on AWS',
                        help='Directory containing PDF chunks')
    parser.add_argument('--output', default='processed_materials/ml_engineering_full',
                        help='Output base directory')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for PDF rendering (default: 150)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from previous run')
    parser.add_argument('--consolidate-only', action='store_true',
                        help='Only consolidate existing outputs, skip OCR processing')
    
    args = parser.parse_args()
    
    processor = BatchOCRProcessor(args.chunks_dir, args.output)
    
    if args.consolidate_only:
        logger.info("Consolidation mode - skipping OCR processing")
        processor.consolidate_output()
    else:
        # Process all chunks
        success, failed = processor.process_all_chunks(
            dpi=args.dpi,
            resume=not args.no_resume
        )
        
        # Consolidate if any chunks were processed
        if success > 0:
            processor.consolidate_output()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
