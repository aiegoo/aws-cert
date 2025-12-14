#!/usr/bin/env python3
"""Crop Kindle screenshots to remove sidebar"""

from PIL import Image
from pathlib import Path
import argparse

def crop_screenshots(input_dir, output_dir, left_crop=300, top_crop=0, right_crop=0, bottom_crop=0):
    """Crop all screenshots to remove margins
    
    Args:
        input_dir: Directory with raw screenshots
        output_dir: Directory for cropped screenshots
        left_crop: Pixels to remove from left (sidebar)
        top_crop: Pixels to remove from top
        right_crop: Pixels to remove from right
        bottom_crop: Pixels to remove from bottom
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
    print()
    
    for i, img_file in enumerate(images, 1):
        if i % 10 == 0:
            print(f"Processing {i}/{len(images)}: {img_file.name}")
        
        # Open image
        img = Image.open(img_file)
        width, height = img.size
        
        # Calculate crop box (left, top, right, bottom)
        crop_box = (
            left_crop,
            top_crop,
            width - right_crop,
            height - bottom_crop
        )
        
        # Crop
        cropped = img.crop(crop_box)
        
        # Save with same name
        output_file = output_path / img_file.name
        cropped.save(output_file, 'PNG')
        
        img.close()
        cropped.close()
    
    print()
    print(f"Complete! Cropped {len(images)} images")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Crop Kindle screenshots')
    parser.add_argument('--input', default='/mnt/c/Users/hsyyu/Documents/kindle_raw',
                       help='Input directory with raw screenshots')
    parser.add_argument('--output', default='/mnt/c/Users/hsyyu/Documents/kindle_cropped',
                       help='Output directory for cropped screenshots')
    parser.add_argument('--left', type=int, default=300,
                       help='Pixels to crop from left (default: 300)')
    parser.add_argument('--top', type=int, default=0,
                       help='Pixels to crop from top (default: 0)')
    parser.add_argument('--right', type=int, default=0,
                       help='Pixels to crop from right (default: 0)')
    parser.add_argument('--bottom', type=int, default=0,
                       help='Pixels to crop from bottom (default: 0)')
    
    args = parser.parse_args()
    
    crop_screenshots(
        args.input,
        args.output,
        args.left,
        args.top,
        args.right,
        args.bottom
    )


if __name__ == '__main__':
    main()
