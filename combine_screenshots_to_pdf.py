#!/usr/bin/env python3
"""
Combine screenshot images into a single PDF
"""

from PIL import Image
import os
import argparse
from pathlib import Path


def combine_images_to_pdf(image_dir, output_pdf, crop_header=0, crop_footer=0, pages_per_pdf=100):
    """Combine images into PDF(s), splitting into chunks if needed
    
    Args:
        image_dir: Directory containing screenshots
        output_pdf: Output PDF filename (or base name for split files)
        crop_header: Pixels to crop from top
        crop_footer: Pixels to crop from bottom
        pages_per_pdf: Maximum pages per PDF file (default: 100)
    """
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(image_dir).glob(ext))
    
    # Sort by filename
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    total_images = len(image_files)
    print(f"Found {total_images} images")
    
    # Calculate number of PDFs needed
    num_pdfs = (total_images + pages_per_pdf - 1) // pages_per_pdf
    
    if num_pdfs > 1:
        print(f"Splitting into {num_pdfs} PDF files ({pages_per_pdf} pages each)")
    
    # Process images in chunks
    output_path = Path(output_pdf)
    output_dir = output_path.parent
    output_base = output_path.stem
    output_ext = output_path.suffix
    
    for pdf_num in range(num_pdfs):
        start_idx = pdf_num * pages_per_pdf
        end_idx = min(start_idx + pages_per_pdf, total_images)
        chunk_files = image_files[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"PDF {pdf_num + 1}/{num_pdfs}: Pages {start_idx + 1}-{end_idx}")
        print(f"{'='*60}")
        
        images = []
        for i, img_path in enumerate(chunk_files, start_idx + 1):
            print(f"Processing {i}/{total_images}: {img_path.name}")
            
            img = Image.open(img_path)
            
            # Crop if specified
            if crop_header or crop_footer:
                width, height = img.size
                bottom = height - crop_footer if crop_footer else height
                img = img.crop((0, crop_header, width, bottom))
            
            # Convert to RGB (PDF requires RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(img)
        
        # Save chunk as PDF
        if images:
            if num_pdfs > 1:
                chunk_pdf = output_dir / f"{output_base}_part{pdf_num + 1:02d}{output_ext}"
            else:
                chunk_pdf = output_path
            
            chunk_pdf.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\nSaving: {chunk_pdf}")
            images[0].save(
                str(chunk_pdf),
                save_all=True,
                append_images=images[1:],
                resolution=100.0,
                quality=85,
                optimize=True
            )
            
            # Get file size
            size_mb = chunk_pdf.stat().st_size / (1024 * 1024)
            print(f"Created: {size_mb:.2f} MB ({len(images)} pages)")
            
            # Clean up images from memory
            for img in images:
                img.close()
    
    print(f"\n{'='*60}")
    print(f"Complete! {total_images} pages processed into {num_pdfs} PDF(s)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Combine screenshots into PDF')
    parser.add_argument('--input-dir', default='/mnt/c/Users/hsyyu/Documents/kindle_screenshots',
                       help='Directory containing screenshots (default: Windows Documents/kindle_screenshots)')
    parser.add_argument('--output', default='output/kindle_book.pdf',
                       help='Output PDF file (default: output/kindle_book.pdf)')
    parser.add_argument('--crop-header', type=int, default=0,
                       help='Pixels to crop from top (default: 0)')
    parser.add_argument('--crop-footer', type=int, default=0,
                       help='Pixels to crop from bottom (default: 0)')
    parser.add_argument('--pages-per-pdf', type=int, default=100,
                       help='Maximum pages per PDF file (default: 100)')
    
    args = parser.parse_args()
    
    # Convert Windows path to WSL path if needed
    input_dir = args.input_dir
    if input_dir.startswith('C:\\') or input_dir.startswith('c:\\'):
        # Convert Windows path to WSL path
        input_dir = input_dir.replace('\\', '/')
        input_dir = '/mnt/c' + input_dir[2:]
    
    combine_images_to_pdf(
        image_dir=input_dir,
        output_pdf=args.output,
        crop_header=args.crop_header,
        crop_footer=args.crop_footer,
        pages_per_pdf=args.pages_per_pdf
    )


if __name__ == '__main__':
    main()
