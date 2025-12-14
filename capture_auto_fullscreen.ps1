# Auto-fullscreen capture - enters F11 mode automatically
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_auto"
)

Write-Host "Auto-Fullscreen Capture" -ForegroundColor Green
Write-Host "=======================" -ForegroundColor Green
Write-Host ""
Write-Host "Make sure Kindle page is open in Chrome, then press ENTER" -ForegroundColor Yellow
Read-Host

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Find Chrome
$chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -like "*Kindle*"} | Select-Object -First 1
if (-not $chrome) {
    $chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -ne ""} | Select-Object -First 1
}

if (-not $chrome) {
    Write-Host "Chrome not found" -ForegroundColor Red
    exit
}

# Add Win32 API
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@

Write-Host "Activating fullscreen mode..." -ForegroundColor Cyan

# Focus Chrome and press F11
[Win32]::SetForegroundWindow($chrome.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 500
[System.Windows.Forms.SendKeys]::SendWait("{F11}")
Start-Sleep -Seconds 2

Write-Host "Starting capture..." -ForegroundColor Green
Write-Host ""

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    # Full screen capture (which is now just Kindle content due to F11)
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $screenshot = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    # Next page
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Complete! Saved to: $OutputDir" -ForegroundColor Green
