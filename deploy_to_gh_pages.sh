#!/bin/bash
# Deploy to GitHub Pages (gh-pages branch)

set -e

echo "🚀 Deploying English study materials to GitHub Pages..."

# Add and commit to master first
echo "📝 Adding files to master..."
git add ml_engineering_english_content.md ml_engineering_english_part2.md ml_engineering_english_part3.md

echo "💾 Committing to master..."
git commit -m "Add English study materials (Modules 0-12, 520 pages reconstructed content)" || echo "Nothing to commit"

echo "🚀 Pushing master to remote..."
git push origin master || echo "Master already up to date"

echo ""
echo "🔄 Switching to gh-pages branch..."
git checkout gh-pages

echo "🔀 Merging master into gh-pages..."
git merge master -m "Merge English study materials from master"

echo "🚀 Pushing to gh-pages (GitHub Pages deployment)..."
git push origin gh-pages

echo "🔙 Returning to master branch..."
git checkout master

echo ""
echo "✅ Successfully deployed to GitHub Pages!"
echo ""
echo "📚 Your materials are available at:"
echo "   https://aiegoo.github.io/aws-cert/"
echo ""
echo "📂 Repository:"
echo "   https://github.com/aiegoo/aws-cert"
echo ""
echo "📄 Files deployed:"
echo "   - ml_engineering_english_content.md (Modules 0-6)"
echo "   - ml_engineering_english_part2.md (Modules 7-9)"  
echo "   - ml_engineering_english_part3.md (Modules 10-12)"
