# Capture Chrome content area only (no address bar/frame)
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_screenshots_content"
)

Write-Host "Chrome Content Area Capture" -ForegroundColor Green
Write-Host "===========================" -ForegroundColor Green
Write-Host ""
Write-Host "Setup:" -ForegroundColor Yellow
Write-Host "1. Open Kindle book in Chrome"
Write-Host "2. Adjust zoom (Ctrl+- or Ctrl++) so full page is visible"
Write-Host "3. Press ENTER to start"
Write-Host ""

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Write-Host "Saving to: $OutputDir" -ForegroundColor Cyan
Read-Host "Press ENTER to start"

Start-Sleep -Seconds 2

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Windows API for getting client area
Add-Type @"
    using System;
    using System.Runtime.InteropServices;
    public class WinAPI {
        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);
        
        [DllImport("user32.dll")]
        public static extern bool GetClientRect(IntPtr hWnd, out RECT lpRect);
        
        [DllImport("user32.dll")]
        public static extern bool ClientToScreen(IntPtr hWnd, ref POINT lpPoint);
    }
    
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
    
    public struct POINT {
        public int X;
        public int Y;
    }
"@

# Find Chrome
Write-Host "Finding Chrome window..." -ForegroundColor Cyan
$chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -like "*Kindle*"} | Select-Object -First 1

if (-not $chrome) {
    $chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -ne ""} | Select-Object -First 1
}

if (-not $chrome) {
    Write-Host "Error: Chrome not found" -ForegroundColor Red
    exit
}

Write-Host "Found: $($chrome.MainWindowTitle)" -ForegroundColor Green
Write-Host ""

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan
    
    # Focus Chrome
    [WinAPI]::SetForegroundWindow($chrome.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 300
    
    # Get client area (content only, no title/address bar)
    $clientRect = New-Object RECT
    [WinAPI]::GetClientRect($chrome.MainWindowHandle, [ref]$clientRect) | Out-Null
    
    # Convert client coordinates to screen coordinates
    $topLeft = New-Object POINT
    $topLeft.X = 0
    $topLeft.Y = 0
    [WinAPI]::ClientToScreen($chrome.MainWindowHandle, [ref]$topLeft) | Out-Null
    
    $width = $clientRect.Right - $clientRect.Left
    $height = $clientRect.Bottom - $clientRect.Top
    
    Write-Host "  Content area: ${width}x${height}" -ForegroundColor DarkGray
    
    # Capture the content area only
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    $screenshot = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($topLeft.X, $topLeft.Y, 0, 0, [System.Drawing.Size]::new($width, $height))
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    Write-Host "  Saved" -ForegroundColor Green
    
    # Next page
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Complete! $MaxPages pages saved to $OutputDir" -ForegroundColor Green
