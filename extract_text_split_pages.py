#!/usr/bin/env python3
"""
Split two-page Kindle spreads into individual pages and extract text
"""

import easyocr
from pathlib import Path
import argparse
from PIL import Image
import io


def split_and_extract_text(image_dir, output_file, crop_header=0, crop_footer=0):
    """Split two-page spreads and extract text from each page
    
    Args:
        image_dir: Directory containing screenshots
        output_file: Output text file
        crop_header: Pixels to crop from top
        crop_footer: Pixels to crop from bottom
    """
    # Initialize EasyOCR reader (English) with GPU acceleration
    print("Initializing OCR reader with GPU...")
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        print("✓ GPU acceleration enabled")
    except Exception as e:
        print(f"GPU not available, using CPU: {e}")
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
    
    print(f"Found {len(image_files)} screenshot images")
    
    # Extract text from each image
    all_text = []
    page_counter = 1
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing screenshot {i}/{len(image_files)}: {img_path.name}")
        
        try:
            # Open image
            img = Image.open(img_path)
            width, height = img.size
            
            print(f"  Image size: {width}x{height}")
            
            # Crop header and footer if specified
            if crop_header or crop_footer:
                bottom = height - crop_footer if crop_footer else height
                img = img.crop((0, crop_header, width, bottom))
                width, height = img.size
                print(f"  After crop: {width}x{height}")
            
            # Detect if it's a two-page spread (wide image)
            # Kindle typically shows two pages side-by-side
            is_spread = width > height * 1.3  # Wide aspect ratio = two pages
            
            if is_spread:
                print(f"  Detected two-page spread, splitting...")
                # Split into left and right pages
                mid_point = width // 2
                
                # Left page
                left_page = img.crop((0, 0, mid_point, height))
                # Right page  
                right_page = img.crop((mid_point, 0, width, height))
                
                pages = [left_page, right_page]
                page_labels = ['Left', 'Right']
            else:
                print(f"  Single page detected")
                pages = [img]
                page_labels = ['']
            
            # OCR each page
            for page_img, label in zip(pages, page_labels):
                # Convert to RGB if needed
                if page_img.mode != 'RGB':
                    page_img = page_img.convert('RGB')
                
                # Save to bytes for EasyOCR
                img_bytes = io.BytesIO()
                page_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Extract text
                print(f"    OCR {label} page {page_counter}...")
                result = reader.readtext(img_bytes.getvalue(), detail=0, paragraph=True)
                
                page_text = '\n'.join(result)
                
                # Add page marker
                all_text.append(f"\n{'='*60}\n")
                all_text.append(f"PAGE {page_counter}{' (' + label + ')' if label else ''}\n")
                all_text.append(f"Screenshot: {img_path.name}\n")
                all_text.append(f"{'='*60}\n\n")
                all_text.append(page_text)
                all_text.append("\n")
                
                print(f"    Extracted {len(page_text)} characters")
                page_counter += 1
            
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            continue
    
    # Save extracted text
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving text to: {output_file}")
    output_path.write_text('\n'.join(all_text), encoding='utf-8')
    
    print(f"\nComplete! Extracted text from {page_counter - 1} pages across {len(image_files)} screenshots")
    print(f"Total characters: {sum(len(t) for t in all_text)}")


def main():
    parser = argparse.ArgumentParser(description='Split two-page spreads and extract text using OCR')
    parser.add_argument('--input-dir', default='/mnt/c/Users/hsyyu/Documents/kindle_screenshots',
                       help='Directory containing screenshots')
    parser.add_argument('--output', default='output/kindle_text_split.txt',
                       help='Output text file (default: output/kindle_text_split.txt)')
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
    
    split_and_extract_text(
        image_dir=input_dir,
        output_file=args.output,
        crop_header=args.crop_header,
        crop_footer=args.crop_footer
    )


if __name__ == '__main__':
    main()
