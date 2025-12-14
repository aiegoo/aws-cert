# Kindle Book Capture for WSL + Windows

Since you're running WSL (Linux) but Chrome is on Windows, we need a two-step process:

## Step 1: Capture Screenshots (Windows PowerShell)

1. **Open PowerShell on Windows** (not in WSL terminal)
2. Navigate to this directory:
   ```powershell
   cd \\wsl$\Ubuntu-22.04\home\aiegoo\repos\aws-cert
   ```

3. Run the capture script:
   ```powershell
   .\capture_kindle_windows.ps1 -MaxPages 500 -WaitTime 2
   ```

4. The script will:
   - Open Chrome with your Kindle book URL
   - Ask you to navigate to the first page
   - Press ENTER to start automated capture
   - Take screenshot, press RIGHT arrow, wait, repeat
   - Save all screenshots to `output/kindle_screenshots/`

**Parameters:**
- `-MaxPages`: Number of pages to capture (default: 500)
- `-WaitTime`: Seconds to wait between pages (default: 2)
- `-Url`: Kindle book URL (default: your current book)

## Step 2: Combine Screenshots into PDF (WSL/Linux)

Back in your WSL terminal, run:

```bash
python3 combine_screenshots_to_pdf.py \
  --input-dir output/kindle_screenshots \
  --output output/kindle_book.pdf \
  --crop-header 100 \
  --crop-footer 50
```

This will:
- Load all screenshots from the directory
- Crop headers/footers if specified
- Combine into a single PDF: `output/kindle_book.pdf`

**Parameters:**
- `--input-dir`: Screenshot directory (default: output/kindle_screenshots)
- `--output`: Output PDF file (default: output/kindle_book.pdf)
- `--crop-header`: Pixels to crop from top (default: 0)
- `--crop-footer`: Pixels to crop from bottom (default: 0)

## Alternative: Manual Process

If automation doesn't work:

### Windows - Manual Screenshots
1. Open Kindle book in Chrome
2. Press `F11` for fullscreen
3. Use screenshot tool: `Win + Shift + S` or Snipping Tool
4. Manually screenshot each page
5. Save to `output/kindle_screenshots/`
6. Press RIGHT arrow for next page

### WSL - Combine to PDF
```bash
python3 combine_screenshots_to_pdf.py
```

## Next Steps

After you have the PDF, we can:
1. Extract text using OCR (EasyOCR or Tesseract)
2. Parse into study materials
3. Create flashcards
4. Deploy to GitHub Pages

## Troubleshooting

**PowerShell Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Can't access WSL files from Windows:**
- In Windows Explorer, go to: `\\wsl$\Ubuntu-22.04\home\aiegoo\repos\aws-cert`
- Or use: `explorer.exe .` from WSL terminal

**Screenshots are too large:**
- Adjust screen resolution before capture
- Or increase crop values to remove more UI elements
