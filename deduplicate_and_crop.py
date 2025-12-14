#!/usr/bin/env python3
"""Deduplicate and crop Kindle screenshots"""

from PIL import Image
import imagehash
from pathlib import Path
import argparse
from collections import defaultdict

def deduplicate_and_crop(input_dir, output_dir, left_crop=300, top_crop=0, right_crop=0, bottom_crop=100, similarity=5):
    """Deduplicate screenshots and crop margins
    
    Args:
        input_dir: Directory with raw screenshots
        output_dir: Directory for unique cropped screenshots
        left_crop: Pixels to crop from left (sidebar)
        top_crop: Pixels to crop from top
        right_crop: Pixels to crop from right
        bottom_crop: Pixels to crop from bottom (cutoff area)
        similarity: Hash difference threshold (lower = more strict)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    images = sorted(input_path.glob("*.png"))
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(images)} screenshots")
    print(f"Crop margins: left={left_crop}, top={top_crop}, right={right_crop}, bottom={bottom_crop}")
    print(f"Deduplicating...")
    print()
    
    # Hash all images to find duplicates
    hashes = {}
    unique_pages = []
    
    for i, img_file in enumerate(images, 1):
        if i % 50 == 0:
            print(f"Hashing {i}/{len(images)}...")
        
        img = Image.open(img_file)
        img_hash = imagehash.average_hash(img)
        img.close()
        
        # Check if similar image already exists
        is_duplicate = False
        for existing_hash, existing_file in hashes.items():
            if abs(img_hash - existing_hash) <= similarity:
                is_duplicate = True
                break
        
        if not is_duplicate:
            hashes[img_hash] = img_file
            unique_pages.append(img_file)
    
    print(f"\nFound {len(unique_pages)} unique pages (removed {len(images) - len(unique_pages)} duplicates)")
    print(f"\nCropping unique pages...")
    print()
    
    # Crop unique images
    for i, img_file in enumerate(unique_pages, 1):
        print(f"Cropping {i}/{len(unique_pages)}: {img_file.name}")
        
        img = Image.open(img_file)
        width, height = img.size
        
        # Crop box
        crop_box = (
            left_crop,
            top_crop,
            width - right_crop,
            height - bottom_crop
        )
        
        cropped = img.crop(crop_box)
        
        # Save with sequential naming
        output_file = output_path / f"page_{i:03d}.png"
        cropped.save(output_file, 'PNG')
        
        img.close()
        cropped.close()
    
    print()
    print(f"Complete! Saved {len(unique_pages)} unique pages")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Deduplicate and crop Kindle screenshots')
    parser.add_argument('--input', default='/mnt/c/Users/hsyyu/Documents/kindle_raw',
                       help='Input directory with raw screenshots')
    parser.add_argument('--output', default='/mnt/c/Users/hsyyu/Documents/kindle_clean',
                       help='Output directory for unique cropped screenshots')
    parser.add_argument('--left', type=int, default=300,
                       help='Pixels to crop from left (sidebar)')
    parser.add_argument('--top', type=int, default=0,
                       help='Pixels to crop from top')
    parser.add_argument('--right', type=int, default=0,
                       help='Pixels to crop from right')
    parser.add_argument('--bottom', type=int, default=100,
                       help='Pixels to crop from bottom (cutoff area)')
    parser.add_argument('--similarity', type=int, default=5,
                       help='Duplicate detection threshold (default: 5)')
    
    args = parser.parse_args()
    
    deduplicate_and_crop(
        args.input,
        args.output,
        args.left,
        args.top,
        args.right,
        args.bottom,
        args.similarity
    )


if __name__ == '__main__':
    main()
