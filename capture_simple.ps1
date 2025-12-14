# Simple screenshot capture - just full screen, no window manipulation
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_screenshots_simple"
)

Write-Host "Simple Screenshot Capture" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT SETUP:" -ForegroundColor Yellow
Write-Host "1. Press F11 in Chrome to enter fullscreen mode"
Write-Host "2. Use Ctrl+- or Ctrl++ to adjust zoom so ENTIRE page is visible"
Write-Host "3. Make sure you can see the full page from top to bottom"
Write-Host "4. Press ENTER here when ready to start capture"
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Write-Host "Saving to: $OutputDir" -ForegroundColor Cyan
Read-Host "Press ENTER to start"

Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan
    
    # Take full screen screenshot
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $screenshot = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    Write-Host "  Saved" -ForegroundColor Green
    
    # Navigate to next page
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Complete! Saved $MaxPages pages" -ForegroundColor Green
