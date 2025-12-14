#!/usr/bin/env python3
"""
Extract text from Kindle screenshots using OCR
"""

import easyocr
from pathlib import Path
import argparse
from PIL import Image
import io


def extract_text_from_screenshots(image_dir, output_file, crop_header=0, crop_footer=0):
    """Extract text from all screenshots using OCR
    
    Args:
        image_dir: Directory containing screenshots
        output_file: Output text file
        crop_header: Pixels to crop from top
        crop_footer: Pixels to crop from bottom
    """
    # Initialize EasyOCR reader (English)
    print("Initializing OCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(image_dir).glob(ext))
    
    # Sort by filename
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Extract text from each image
    all_text = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing {i}/{len(image_files)}: {img_path.name}")
        
        try:
            # Open and crop image
            img = Image.open(img_path)
            
            if crop_header or crop_footer:
                width, height = img.size
                bottom = height - crop_footer if crop_footer else height
                img = img.crop((0, crop_header, width, bottom))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes for EasyOCR
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Extract text
            result = reader.readtext(img_bytes.getvalue(), detail=0, paragraph=True)
            
            page_text = '\n'.join(result)
            
            # Add page marker
            all_text.append(f"\n{'='*60}\n")
            all_text.append(f"PAGE {i}\n")
            all_text.append(f"{'='*60}\n\n")
            all_text.append(page_text)
            all_text.append("\n")
            
            print(f"  Extracted {len(page_text)} characters")
            
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            continue
    
    # Save extracted text
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving text to: {output_file}")
    output_path.write_text('\n'.join(all_text), encoding='utf-8')
    
    print(f"\nComplete! Extracted text from {len(image_files)} pages")
    print(f"Total characters: {sum(len(t) for t in all_text)}")


def main():
    parser = argparse.ArgumentParser(description='Extract text from screenshots using OCR')
    parser.add_argument('--input-dir', default='/mnt/c/Users/hsyyu/Documents/kindle_screenshots',
                       help='Directory containing screenshots')
    parser.add_argument('--output', default='output/kindle_text.txt',
                       help='Output text file (default: output/kindle_text.txt)')
    parser.add_argument('--crop-header', type=int, default=100,
                       help='Pixels to crop from top (default: 100)')
    parser.add_argument('--crop-footer', type=int, default=50,
                       help='Pixels to crop from bottom (default: 50)')
    
    args = parser.parse_args()
    
    # Convert Windows path to WSL path if needed
    input_dir = args.input_dir
    if input_dir.startswith('C:\\') or input_dir.startswith('c:\\'):
        input_dir = input_dir.replace('\\', '/')
        input_dir = '/mnt/c' + input_dir[2:]
    
    extract_text_from_screenshots(
        image_dir=input_dir,
        output_file=args.output,
        crop_header=args.crop_header,
        crop_footer=args.crop_footer
    )


if __name__ == '__main__':
    main()
