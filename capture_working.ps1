# Hide PowerShell and focus Kindle before capture
param(
    [int]$MaxPages = 560,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_working"
)

Write-Host "Kindle Capture - PowerShell Hidden" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "Setup:" -ForegroundColor Yellow
Write-Host "1. Press F11 in Kindle to enter fullscreen" -ForegroundColor Gray
Write-Host "2. Zoom so full page is visible" -ForegroundColor Gray
Write-Host "3. Press ENTER here" -ForegroundColor Gray
Write-Host "4. PowerShell will hide itself and capture Kindle" -ForegroundColor Gray
Write-Host ""

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Read-Host "Ready? Press ENTER"

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Windows API
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int nIndex);
    [DllImport("kernel32.dll")]
    public static extern IntPtr GetConsoleWindow();
}
"@

# Set DPI awareness
[Win32]::SetProcessDPIAware() | Out-Null

# Get actual screen size
$width = [Win32]::GetSystemMetrics(0)
$height = [Win32]::GetSystemMetrics(1)

Write-Host "Screen: ${width}x${height}" -ForegroundColor Cyan

# Find Chrome/Kindle
$chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -like "*Kindle*"} | Select-Object -First 1
if (-not $chrome) {
    $chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -ne ""} | Select-Object -First 1
}

if (-not $chrome) {
    Write-Host "Chrome not found!" -ForegroundColor Red
    exit
}

Write-Host "Found: $($chrome.MainWindowTitle)" -ForegroundColor Green
Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Yellow
Write-Host "PowerShell will minimize itself..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Hide PowerShell window
$consoleWindow = [Win32]::GetConsoleWindow()
[Win32]::ShowWindow($consoleWindow, 6) | Out-Null  # SW_MINIMIZE

Start-Sleep -Milliseconds 500

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    # Focus Kindle
    [Win32]::SetForegroundWindow($chrome.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 300
    
    # Capture
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
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

# Restore PowerShell window
[Win32]::ShowWindow($consoleWindow, 9) | Out-Null  # SW_RESTORE

Write-Host ""
Write-Host "Complete! Saved to: $OutputDir" -ForegroundColor Green
