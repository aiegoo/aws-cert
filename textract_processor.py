#!/usr/bin/env python3
"""
AWS Textract PDF Processor
Processes PDF chunks using AWS Textract for high-quality OCR
"""

import boto3
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

class TextractProcessor:
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """Initialize Textract processor"""
        self.s3_client = boto3.client('s3', region_name=region)
        self.textract_client = boto3.client('textract', region_name=region)
        self.bucket_name = bucket_name
        self.region = region
        
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Bucket {self.bucket_name} exists")
        except:
            print(f"Creating bucket {self.bucket_name}...")
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            print(f"✓ Created bucket {self.bucket_name}")
    
    def upload_pdf(self, pdf_path: Path, s3_key: str) -> str:
        """Upload PDF to S3"""
        print(f"Uploading {pdf_path.name} to S3...")
        self.s3_client.upload_file(str(pdf_path), self.bucket_name, s3_key)
        return s3_key
    
    def process_document(self, s3_key: str) -> Dict:
        """Process document with Textract (synchronous for <2000 pages)"""
        print(f"Processing {s3_key} with Textract...")
        
        response = self.textract_client.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': self.bucket_name,
                    'Name': s3_key
                }
            }
        )
        
        job_id = response['JobId']
        print(f"  Job ID: {job_id}")
        
        # Wait for job to complete
        while True:
            response = self.textract_client.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                print(f"  ✓ Job completed")
                break
            elif status == 'FAILED':
                raise Exception(f"Textract job failed: {response.get('StatusMessage', 'Unknown error')}")
            else:
                print(f"  Status: {status}, waiting...")
                time.sleep(5)
        
        # Get all results (handle pagination)
        blocks = response['Blocks']
        next_token = response.get('NextToken')
        
        while next_token:
            response = self.textract_client.get_document_text_detection(
                JobId=job_id,
                NextToken=next_token
            )
            blocks.extend(response['Blocks'])
            next_token = response.get('NextToken')
        
        return self.extract_text_from_blocks(blocks)
    
    def extract_text_from_blocks(self, blocks: List[Dict]) -> Dict:
        """Extract text from Textract blocks"""
        pages = {}
        
        for block in blocks:
            if block['BlockType'] == 'LINE':
                page_num = block['Page']
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append({
                    'text': block['Text'],
                    'confidence': block['Confidence']
                })
        
        # Format results
        results = {
            'total_pages': len(pages),
            'pages': []
        }
        
        for page_num in sorted(pages.keys()):
            page_lines = pages[page_num]
            page_text = '\n'.join([line['text'] for line in page_lines])
            avg_conf = sum(line['confidence'] for line in page_lines) / len(page_lines)
            
            results['pages'].append({
                'page_num': page_num,
                'text': page_text,
                'confidence': avg_conf,
                'lines': page_lines
            })
        
        return results
    
    def process_chunk(self, chunk_path: Path, output_dir: Path, chunk_num: int):
        """Process a single PDF chunk"""
        print(f"\n{'='*60}")
        print(f"Processing Chunk {chunk_num:03d}: {chunk_path.name}")
        print(f"{'='*60}")
        
        # Upload to S3
        s3_key = f"textract-temp/{chunk_path.name}"
        self.upload_pdf(chunk_path, s3_key)
        
        # Process with Textract
        results = self.process_document(s3_key)
        
        # Save results
        output_file = output_dir / f"chunk_{chunk_num:03d}_textract.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunk_num': chunk_num,
                'source_file': chunk_path.name,
                'data': results
            }, f, ensure_ascii=False, indent=2)
        
        # Cleanup S3
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
        
        print(f"✓ Saved to {output_file}")
        print(f"  Pages: {results['total_pages']}")
        print(f"  Avg confidence: {sum(p['confidence'] for p in results['pages']) / len(results['pages']):.1f}%")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PDFs with AWS Textract')
    parser.add_argument('chunks_dir', help='Directory containing PDF chunks')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--bucket', default='textract-ocr-temp-bucket', help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--start', type=int, default=1, help='Start chunk number')
    parser.add_argument('--end', type=int, help='End chunk number')
    
    args = parser.parse_args()
    
    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)
    
    # Get chunk files
    chunk_files = sorted(chunks_dir.glob('*_chunk_*.pdf'))
    
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        sys.exit(1)
    
    print(f"Found {len(chunk_files)} chunks")
    
    # Filter by range
    if args.end:
        chunk_files = [f for f in chunk_files if args.start <= int(f.stem.split('_chunk_')[1].split('_')[0]) <= args.end]
    else:
        chunk_files = [f for f in chunk_files if int(f.stem.split('_chunk_')[1].split('_')[0]) >= args.start]
    
    print(f"Processing {len(chunk_files)} chunks (from {args.start})")
    
    # Initialize processor
    processor = TextractProcessor(args.bucket, args.region)
    processor.create_bucket_if_not_exists()
    
    # Process each chunk
    for chunk_path in chunk_files:
        chunk_num = int(chunk_path.stem.split('_chunk_')[1].split('_')[0])
        try:
            processor.process_chunk(chunk_path, output_dir, chunk_num)
        except Exception as e:
            print(f"✗ Error processing chunk {chunk_num}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
