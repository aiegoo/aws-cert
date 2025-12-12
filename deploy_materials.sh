#!/bin/bash
# Deploy English study materials to GitHub

set -e

echo "🚀 Deploying English study materials..."

# Add the new files
git add ml_engineering_english_content.md
git add ml_engineering_english_part2.md
git add ml_engineering_english_part3.md

# Commit
git commit -m "Add comprehensive English study guide for AWS ML Engineering

- Complete coverage of Modules 0-12 (520 pages)
- Reconstructed from OCR structure + AWS ML curriculum knowledge
- Includes code examples, best practices, exam preparation
- Ready for MLA-C01 certification study"

# Push to remote
git push origin master

echo "✅ Successfully deployed to GitHub!"
echo ""
echo "📚 View your study materials:"
echo "   https://github.com/aiegoo/aws-cert"
echo ""
echo "Files deployed:"
echo "   - ml_engineering_english_content.md (Modules 0-6)"
echo "   - ml_engineering_english_part2.md (Modules 7-9)"
echo "   - ml_engineering_english_part3.md (Modules 10-12)"
