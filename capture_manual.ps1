# Manual region capture - you specify the coordinates
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_screenshots_manual"
)

Write-Host "Manual Region Capture" -ForegroundColor Green
Write-Host "=====================" -ForegroundColor Green
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Yellow
Write-Host "1. Open Kindle and note the position and size of content area"
Write-Host "2. Use Windows Snipping Tool (Win+Shift+S) to measure:"
Write-Host "   - Top-left corner position (X, Y)"
Write-Host "   - Width and Height of content area"
Write-Host "3. Enter those values below"
Write-Host ""

# Get region from user
$x = Read-Host "Enter X position (left edge, e.g., 100)"
$y = Read-Host "Enter Y position (top edge, e.g., 150)"
$width = Read-Host "Enter Width (e.g., 1200)"
$height = Read-Host "Enter Height (e.g., 900)"

Write-Host ""
Write-Host "Capture region:" -ForegroundColor Cyan
Write-Host "  Position: ($x, $y)" -ForegroundColor Gray
Write-Host "  Size: ${width}x${height}" -ForegroundColor Gray
Write-Host ""

$confirm = Read-Host "Is this correct? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled" -ForegroundColor Yellow
    exit
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    # Capture the region
    $screenshot = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($x, $y, 0, 0, [System.Drawing.Size]::new($width, $height))
    
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
Write-Host "Complete! $MaxPages pages saved to:" -ForegroundColor Green
Write-Host "$OutputDir" -ForegroundColor Cyan
