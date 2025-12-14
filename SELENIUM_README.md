# Browser Text Extractor - Selenium Automation

Three Python scripts for extracting text and screenshots from web pages using Selenium WebDriver.

## Installation

```bash
pip install -r requirements_selenium.txt
```

You also need Chrome browser and ChromeDriver installed. On Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y chromium-browser chromium-chromedriver
```

## Scripts Overview

1. **`page_capture_extractor.py`** - Page flipping automation (kindleOCRer technique)
   - Capture e-books, presentations, documents as PDF or images
   - Navigate using arrow keys or Next button
   - Automatic duplicate page detection
   - Session persistence (stay logged in)

2. **`browser_text_extractor.py`** - Command-line text extraction
   - Extract text, headings, code blocks, tables
   - CSS selector support
   - Headless mode

3. **`interactive_browser_extractor.py`** - Interactive shell
   - Manual navigation with extraction commands
   - Real-time element highlighting

## Usage

### 1. Page Capture Extractor (NEW - kindleOCRer technique)

**Capture Kindle Cloud Reader book as PDF:**
```bash
# First, open Chrome and login to read.amazon.com manually
# Find your Chrome profile directory:
#   Linux: ~/.config/google-chrome/
#   Mac: ~/Library/Application Support/Google/Chrome/
#   Windows: C:\Users\<user>\AppData\Local\Google\Chrome\User Data

python page_capture_extractor.py https://read.amazon.com/reader \
  --user-data-dir ~/.config/google-chrome \
  --profile-dir Default \
  --capture-screenshots \
  --arrow-key right \
  --max-pages 300 \
  --crop-header 100 \
  --crop-footer 50 \
  --fullscreen \
  --save-pdf output/my_book.pdf
```

**Capture Google Slides presentation:**
```bash
python page_capture_extractor.py <slides-presentation-url> \
  --capture-screenshots \
  --arrow-key down \
  --save-pdf presentation.pdf \
  --save-images slides_images/
```

**Extract text from paginated document:**
```bash
python page_capture_extractor.py <document-url> \
  --extract-text \
  --arrow-key pagedown \
  --max-pages 50 \
  --output document.txt \
  --copy
```

**Capture with Next button (instead of arrow keys):**
```bash
python page_capture_extractor.py <url> \
  --capture-screenshots \
  --next-button "#next-page-btn" \
  --save-pdf output.pdf
```

**Key Features:**
- `--user-data-dir` - Use existing Chrome profile (stay logged in)
- `--arrow-key` - Navigate with right/down/pagedown keys
- `--next-button` - Alternative: click Next button
- `--crop-header/footer` - Remove navigation bars from screenshots
- `--fullscreen` - Maximize capture area
- `--max-pages` - Safety limit
- Automatic duplicate detection (stops at end of book)

### 2. Command-Line Text Extractor

**Basic text extraction:**
```bash
# Extract all text
python browser_text_extractor.py https://example.com --copy

# Extract specific element
python browser_text_extractor.py https://example.com -s "#main-content" -o output.txt

# Extract headings only
python browser_text_extractor.py https://example.com --headings --copy

# Extract code blocks
python browser_text_extractor.py https://example.com --code -o code.txt
```

### 3. Interactive Browser

Manual navigation with extraction commands:

Manual navigation with extraction commands:

```bash
python interactive_browser_extractor.py
```

**Interactive Commands:**
```
copy / c          - Copy all page text
main / m          - Copy main content
headings / h      - Copy all headings
id <id>           - Copy element by ID
class <name>      - Copy elements by class
css <selector>    - Copy element by CSS selector
highlight <sel>   - Highlight element (for finding selectors)
url               - Show current URL
save              - Save page HTML source
goto <url>        - Navigate to URL
back              - Go back
forward           - Go forward
refresh           - Refresh page
quit / q          - Exit
```

**Example Session:**
```bash
$ python interactive_browser_extractor.py
Enter starting URL: https://aiegoo.github.io/aws-cert/study_guide.html

>>> main          # Copy main content
✓ Copied main content from 'article': 45231 characters

>>> headings      # Copy all headings
✓ Copied 87 headings to clipboard

>>> css .module   # Copy specific section
✓ Copied element '.module': 3421 characters

>>> quit          # Exit
```

## Examples

### Extract Study Guide Content
```bash
# Get your study guide
python browser_text_extractor.py \
  https://aiegoo.github.io/aws-cert/study_guide.html \
  --selector "article" \
  --copy \
  -o study_guide_extracted.txt

# Extract only module headings
python browser_text_extractor.py \
  https://aiegoo.github.io/aws-cert/study_guide.html \
  --headings \
  -o module_headings.txt
```

### Extract Code Examples
```bash
# Get all code blocks from documentation
python browser_text_extractor.py \
  https://docs.aws.amazon.com/sagemaker/latest/dg/example.html \
  --code \
  --copy
```

### Extract PDF Course Materials (if rendered in browser)
```bash
# Some PDFs render as HTML in browser
python browser_text_extractor.py \
  file:///home/user/document.pdf \
  --scroll \
  --full-page \
  -o pdf_extracted.txt
```

## Tips

1. **Finding CSS Selectors:**
   - Use browser DevTools (F12 → Elements → Right-click → Copy selector)
   - Use `highlight` command in interactive mode
   - Common selectors: `#id`, `.class`, `tag`, `[attribute]`

2. **Handling Dynamic Content:**
   - Use `--scroll` to load lazy content
   - Wait time can be adjusted in the code (default: 2 seconds)

3. **Clipboard Access:**
   - Linux: requires `xclip` or `xsel` (`sudo apt-get install xclip`)
   - macOS: works natively
   - Windows: works natively

4. **Headless Mode:**
   - Faster and uses less resources
   - Good for automated scripts
   - Can't see what's happening (harder to debug)

## Troubleshooting

**ChromeDriver not found:**
```bash
# Install via webdriver-manager (automatic)
pip install webdriver-manager

# Or manually download from:
# https://chromedriver.chromium.org/
```

**Can't copy to clipboard:**
```bash
# Linux: install xclip
sudo apt-get install xclip

# Test with:
python -c "import pyperclip; pyperclip.copy('test'); print(pyperclip.paste())"
```

**Element not found:**
- Page may not be fully loaded (increase wait time in code)
- JavaScript may be building content (use `--scroll`)
- Selector may be incorrect (use browser DevTools to verify)

## Advanced: Customization

Edit the scripts to:
- Change wait times (default: 2 seconds)
- Add authentication (login automation)
- Handle popups/dialogs
- Extract specific data patterns
- Add custom parsing logic
