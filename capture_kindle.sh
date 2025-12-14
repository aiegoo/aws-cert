#!/bin/bash
# Capture Kindle book that's already open in Chrome
# Make sure you're on the FIRST page you want to capture before running this

echo "=== Kindle Book Capture ==="
echo ""
echo "IMPORTANT: Before running, make sure:"
echo "1. Chrome is open with the Kindle book"
echo "2. You're on the FIRST page you want to capture"
echo "3. Book is in fullscreen reading mode"
echo ""
read -p "Ready? Press Enter to start (Ctrl+C to cancel)..."

# Close Chrome first to get the profile
echo ""
echo "Please CLOSE Chrome now, then press Enter..."
read

# Find Chrome user data directory
CHROME_DIR=""
if [ -d "$HOME/.config/google-chrome" ]; then
    CHROME_DIR="$HOME/.config/google-chrome"
elif [ -d "$HOME/.config/chromium" ]; then
    CHROME_DIR="$HOME/.config/chromium"
elif [ -d "$HOME/Library/Application Support/Google/Chrome" ]; then
    CHROME_DIR="$HOME/Library/Application Support/Google/Chrome"
fi

if [ -z "$CHROME_DIR" ]; then
    echo "Error: Chrome profile not found"
    exit 1
fi

echo "Found Chrome directory: $CHROME_DIR"
echo ""

# Get book details
read -p "Book title (for filename): " BOOK_TITLE
read -p "Max pages to capture (default 500): " MAX_PAGES
MAX_PAGES=${MAX_PAGES:-500}

read -p "Crop header pixels (default 100): " CROP_HEADER
CROP_HEADER=${CROP_HEADER:-100}

read -p "Crop footer pixels (default 50): " CROP_FOOTER
CROP_FOOTER=${CROP_FOOTER:-50}

# Sanitize filename
FILENAME=$(echo "$BOOK_TITLE" | sed 's/[^a-zA-Z0-9]/_/g' | tr '[:upper:]' '[:lower:]')

echo ""
echo "=== Capture Settings ==="
echo "Title: $BOOK_TITLE"
echo "Output: kindle_${FILENAME}.pdf"
echo "Max pages: $MAX_PAGES"
echo "Crop: header=${CROP_HEADER}px, footer=${CROP_FOOTER}px"
echo ""

read -p "Start capture? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Run capture
echo ""
echo "Starting capture... Browser will open automatically"
echo "Capturing pages with RIGHT arrow key"
echo ""

python3 page_capture_extractor.py \
  "https://read.amazon.com/?asin=B0FG1YNWST&ref_=kwl_kr_iv_rec_2" \
  --user-data-dir "$CHROME_DIR" \
  --profile-dir "Default" \
  --capture-screenshots \
  --arrow-key right \
  --max-pages $MAX_PAGES \
  --crop-header $CROP_HEADER \
  --crop-footer $CROP_FOOTER \
  --fullscreen \
  --wait-time 0.7 \
  --save-pdf "output/kindle_${FILENAME}.pdf" \
  --save-images "output/kindle_${FILENAME}_images"

echo ""
echo "=== Capture Complete ==="
echo "PDF saved to: output/kindle_${FILENAME}.pdf"
echo "Images saved to: output/kindle_${FILENAME}_images/"
echo ""
echo "Next steps:"
echo "1. Review the PDF"
echo "2. Extract text: python extract_kindle_text.py output/kindle_${FILENAME}.pdf"
echo "3. Generate study guide: python generate_study_guide.py"
