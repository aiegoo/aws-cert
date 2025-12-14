# Simple fullscreen capture - assumes you pressed F11 already
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_final"
)

Write-Host "Kindle Screenshot Capture" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""
Write-Host "BEFORE starting:" -ForegroundColor Yellow
Write-Host "1. Open Kindle to first page" -ForegroundColor Gray
Write-Host "2. Press F11 to enter fullscreen" -ForegroundColor Gray
Write-Host "3. Adjust zoom (Ctrl+/Ctrl-) if needed so full page is visible" -ForegroundColor Gray
Write-Host "4. Press ENTER here to start" -ForegroundColor Gray
Write-Host ""

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Read-Host "Ready? Press ENTER"

Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Capturing page $i / $MaxPages..." -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    # Full screen capture
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $screenshot = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    # Navigate to next page
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Complete! $MaxPages pages saved to:" -ForegroundColor Green
Write-Host "$OutputDir" -ForegroundColor Cyan
