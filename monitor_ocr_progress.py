#!/usr/bin/env python3
"""
Monitor OCR batch processing progress
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime


def format_duration(seconds):
    """Format seconds into human-readable duration"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    progress_file = Path('processed_materials/ml_engineering_full/processing_progress.json')
    log_file = Path('batch_ocr_processing.log')
    
    if not progress_file.exists():
        print("❌ Progress file not found. Processing may not have started yet.")
        print(f"Looking for: {progress_file}")
        return 1
    
    # Load progress
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    completed = progress.get('completed_chunks', [])
    failed = progress.get('failed_chunks', [])
    last_updated = progress.get('last_updated')
    
    total_chunks = 26
    completed_count = len(completed)
    failed_count = len(failed)
    remaining = total_chunks - completed_count - failed_count
    
    # Calculate progress percentage
    progress_pct = (completed_count / total_chunks) * 100
    
    # Display status
    print("=" * 60)
    print("ML Engineering OCR Processing Status")
    print("=" * 60)
    print(f"📊 Progress: {completed_count}/{total_chunks} chunks ({progress_pct:.1f}%)")
    print(f"✅ Completed: {completed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏳ Remaining: {remaining}")
    
    if last_updated:
        print(f"🕒 Last Update: {last_updated}")
    
    # Progress bar
    bar_width = 50
    filled = int(bar_width * progress_pct / 100)
    bar = '█' * filled + '░' * (bar_width - filled)
    print(f"\n[{bar}] {progress_pct:.1f}%\n")
    
    # Show recently completed chunks
    if completed:
        print("Recently completed chunks:")
        for chunk in completed[-5:]:
            print(f"  ✓ {chunk}")
    
    # Show failed chunks if any
    if failed:
        print("\nFailed chunks:")
        for chunk in failed:
            print(f"  ✗ {chunk}")
    
    # Show log tail
    if log_file.exists():
        print("\n" + "=" * 60)
        print("Recent Log Entries:")
        print("=" * 60)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    
    # Estimate remaining time (rough estimate: 20 pages/hour on CPU)
    if completed_count > 0 and remaining > 0:
        # Rough estimate: ~30 seconds per page average
        est_minutes = (remaining * 20 * 30) / 60
        print(f"\n⏱️  Estimated time remaining: ~{int(est_minutes)} minutes")
    
    print("\n" + "=" * 60)
    
    # Check if processing is complete
    if completed_count == total_chunks:
        print("🎉 Processing Complete!")
        print("\nNext steps:")
        print("  1. Review output: processed_materials/ml_engineering_full/")
        print("  2. Generate study materials:")
        print("     python3 generate_study_materials.py")
        print("=" * 60)
    elif remaining > 0:
        print("Processing is still running...")
        print("\nMonitor progress:")
        print("  watch -n 10 python3 monitor_ocr_progress.py")
        print("\nOr check log:")
        print("  tail -f batch_ocr_processing.log")
        print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
