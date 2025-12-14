# High-DPI aware capture
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_highdpi"
)

Write-Host "High-DPI Aware Capture" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host ""
Write-Host "Press F11 in Kindle and zoom so full page is visible" -ForegroundColor Yellow
Write-Host "Then press ENTER" -ForegroundColor Yellow
Read-Host

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Set DPI awareness
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class DPI {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
}
"@
[DPI]::SetProcessDPIAware() | Out-Null

Write-Host "Starting in 2 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 2

# Get actual screen dimensions with DPI scaling
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Screen {
    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int nIndex);
}
"@

$width = [Screen]::GetSystemMetrics(0)   # SM_CXSCREEN
$height = [Screen]::GetSystemMetrics(1)  # SM_CYSCREEN

Write-Host "Actual screen size: ${width}x${height}" -ForegroundColor Cyan
Write-Host ""

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    # Capture with correct dimensions
    $screenshot = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen(0, 0, 0, 0, [System.Drawing.Size]::new($width, $height))
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    # Next page
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Complete! Saved to: $OutputDir" -ForegroundColor Green
