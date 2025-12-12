#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/aiegoo/repos/aws-cert')
from batch_ocr_processor import BatchOCRProcessor

processor = BatchOCRProcessor(
    'chunks_ML Engineering on AWS',
    'processed_materials/ml_engineering_full'
)
result = processor.consolidate_output()
print(f"Consolidated {len(result['chunks'])} chunks successfully!")
