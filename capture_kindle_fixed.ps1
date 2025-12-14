# PowerShell script to capture Kindle book screenshots on Windows
# Run this from PowerShell on Windows (not WSL)

param(
    [string]$Url = "https://read.amazon.com/?asin=B0FG1YNWST&ref_=kwl_kr_iv_rec_2",
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_screenshots"
)

Write-Host "Kindle Book Screenshot Capture" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Yellow
Write-Host "1. This script will open Chrome with the Kindle book URL"
Write-Host "2. Log in to your Amazon account if needed"
Write-Host "3. Navigate to the first page you want to capture"
Write-Host "4. Press ENTER in this PowerShell window to start capture"
Write-Host "5. The script will automatically:"
Write-Host "   - Take a screenshot"
Write-Host "   - Press RIGHT arrow key to go to next page"
Write-Host "   - Repeat until $MaxPages pages are captured"
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Write-Host "Screenshots will be saved to: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Open Chrome (user should already have it open and logged in)
Write-Host "Opening Chrome..." -ForegroundColor Cyan
Start-Process "chrome.exe" $Url

Write-Host ""
Write-Host "Please navigate to the first page you want to capture in Chrome,"
Write-Host "then press ENTER to start automated capture..." -ForegroundColor Yellow
Read-Host

Write-Host ""
Write-Host "Starting capture in 3 seconds..." -ForegroundColor Green
Write-Host "TIP: Resize Chrome window to show full page before starting" -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Load Windows Forms for SendKeys
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Add type for window management
Add-Type @"
    using System;
    using System.Runtime.InteropServices;
    public class Win32 {
        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);
        [DllImport("user32.dll")]
        public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
        [DllImport("user32.dll")]
        public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    }
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
"@

# Find Chrome window
Write-Host "Finding Chrome window with Kindle book..." -ForegroundColor Cyan
$chromeProcess = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -like "*Kindle*"} | Select-Object -First 1

if (-not $chromeProcess) {
    Write-Host "Warning: Could not find Chrome with Kindle. Looking for any Chrome window..." -ForegroundColor Yellow
    $chromeProcess = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -ne ""} | Select-Object -First 1
}

if ($chromeProcess) {
    Write-Host "Found Chrome window: $($chromeProcess.MainWindowTitle)" -ForegroundColor Green
} else {
    Write-Host "Error: Could not find Chrome window. Please ensure Chrome is open with the Kindle book." -ForegroundColor Red
    exit
}

# Zoom out on first page to fit content
Write-Host "Zooming out to fit content..." -ForegroundColor Cyan
[Win32]::SetForegroundWindow($chromeProcess.MainWindowHandle)
Start-Sleep -Milliseconds 500
[System.Windows.Forms.SendKeys]::SendWait("^0")  # Reset zoom
Start-Sleep -Milliseconds 200
[System.Windows.Forms.SendKeys]::SendWait("^{-}")  # Zoom out
Start-Sleep -Milliseconds 100
[System.Windows.Forms.SendKeys]::SendWait("^{-}")  # Zoom out
Start-Sleep -Milliseconds 100
[System.Windows.Forms.SendKeys]::SendWait("^{-}")  # Zoom out
Start-Sleep -Milliseconds 500

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Capturing page $i / $MaxPages..." -ForegroundColor Cyan
    
    # Bring Chrome to foreground
    [Win32]::SetForegroundWindow($chromeProcess.MainWindowHandle)
    Start-Sleep -Milliseconds 300
    
    # Take screenshot
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    try {
        # Get Chrome window rectangle
        $rect = New-Object RECT
        [Win32]::GetWindowRect($chromeProcess.MainWindowHandle, [ref]$rect) | Out-Null
        
        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        
        Write-Host "  Window: ${width}x${height} at ($($rect.Left), $($rect.Top))" -ForegroundColor DarkGray
        
        # Capture the specific window area
        $screenshot = New-Object System.Drawing.Bitmap $width, $height
        $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, [System.Drawing.Size]::new($width, $height))
        
        $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $screenshot.Dispose()
        
        Write-Host "  Saved successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "  Error: $_" -ForegroundColor Red
        continue
    }
    
    # Press RIGHT arrow to go to next page
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    
    # Wait for page to load
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Capture complete! $MaxPages pages saved to $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review screenshots in: $OutputDir"
Write-Host "2. From WSL, run OCR to extract text:"
Write-Host "   python3 extract_text_split_pages.py"
