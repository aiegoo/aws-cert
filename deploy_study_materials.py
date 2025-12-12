#!/usr/bin/env python3
"""Deploy English study materials to GitHub Pages"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, description=""):
    """Run command and return result"""
    if description:
        print(f"\n{description}")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def main():
    # Check if files exist
    files = ['ml_engineering_english_content.md', 'ml_engineering_english_part2.md', 'ml_engineering_english_part3.md']
    
    print("📝 Checking for study material files...")
    for f in files:
        if not Path(f).exists():
            print(f"❌ File not found: {f}")
            return 1
        else:
            size = Path(f).stat().st_size / 1024 / 1024  # MB
            print(f"✓ {f} ({size:.1f} MB)")
    
    # Add files
    run_cmd(['git', 'add'] + files, "📝 Adding files to git...")
    
    # Check status
    result = run_cmd(['git', 'status', '--short'], "📊 Git status:")
    
    if not result.stdout.strip():
        print("\n⚠️  No changes to commit - files might already be committed")
        print("Proceeding with merge to gh-pages anyway...")
    else:
        # Commit
        result = run_cmd(['git', 'commit', '-m', 'Add comprehensive English study materials for AWS ML Engineering (Modules 0-12)'], 
                        "💾 Committing...")
        
        if result.returncode != 0:
            print("❌ Commit failed")
            return 1
        
        # Push to master
        result = run_cmd(['git', 'push', 'origin', 'master'], "🚀 Pushing to master...")
        if result.returncode != 0:
            print("❌ Push to master failed")
            return 1
    
    # Checkout gh-pages
    result = run_cmd(['git', 'checkout', 'gh-pages'], "🔄 Switching to gh-pages...")
    if result.returncode != 0:
        print("❌ Failed to checkout gh-pages")
        return 1
    
    # Merge master
    result = run_cmd(['git', 'merge', 'master', '-m', 'Merge English study materials from master'], 
                    "🔀 Merging master into gh-pages...")
    
    # Push gh-pages
    result = run_cmd(['git', 'push', 'origin', 'gh-pages'], "🚀 Pushing to gh-pages (GitHub Pages)...")
    if result.returncode != 0:
        print("❌ Push to gh-pages failed")
        run_cmd(['git', 'checkout', 'master'], "Returning to master...")
        return 1
    
    # Return to master
    run_cmd(['git', 'checkout', 'master'], "🔙 Returning to master...")
    
    print("\n" + "="*60)
    print("✅ Successfully deployed to GitHub Pages!")
    print("="*60)
    print("\n📚 Your English study materials are now available at:")
    print("   https://aiegoo.github.io/aws-cert/")
    print("\n📂 GitHub Repository:")
    print("   https://github.com/aiegoo/aws-cert")
    print("\n📄 Deployed files:")
    for f in files:
        print(f"   ✓ {f}")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
