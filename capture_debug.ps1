# Debug version - shows detailed info
param(
    [int]$MaxPages = 5,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_debug"
)

Write-Host "Debug Capture Test" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""
Write-Host "Output directory will be: $OutputDir" -ForegroundColor Cyan
Write-Host ""

try {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Host "Directory created successfully" -ForegroundColor Green
} catch {
    Write-Host "ERROR creating directory: $_" -ForegroundColor Red
    exit
}

Write-Host "Press F11 in Kindle, then press ENTER" -ForegroundColor Yellow
Read-Host

Write-Host ""
Write-Host "Starting in 2 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 2

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Test capture
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host ""
    Write-Host "=== Page $i ===" -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    Write-Host "Filename: $filename" -ForegroundColor Gray
    
    try {
        # Capture
        $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        Write-Host "Screen size: $($bounds.Width)x$($bounds.Height)" -ForegroundColor Gray
        
        $screenshot = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
        $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
        $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        
        $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $screenshot.Dispose()
        
        # Check if file exists
        if (Test-Path $filename) {
            $fileSize = (Get-Item $filename).Length / 1KB
            Write-Host "SUCCESS! File saved: $([math]::Round($fileSize, 2)) KB" -ForegroundColor Green
        } else {
            Write-Host "ERROR! File not found after save" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR capturing: $_" -ForegroundColor Red
    }
    
    # Next page
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Test complete! Check: $OutputDir" -ForegroundColor Green
