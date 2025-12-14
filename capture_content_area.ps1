# Capture Kindle content area only (exclude sidebar)
param(
    [int]$MaxPages = 560,
    [int]$WaitTime = 2,
    [int]$LeftMargin = 300,    # Pixels to skip from left (sidebar)
    [int]$TopMargin = 0,       # Pixels to skip from top
    [int]$RightMargin = 0,     # Pixels to skip from right
    [int]$BottomMargin = 0,    # Pixels to skip from bottom
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_content_only"
)

Write-Host "Kindle Content Area Capture" -ForegroundColor Green
Write-Host "===========================" -ForegroundColor Green
Write-Host ""
Write-Host "This will capture ONLY the book content, excluding sidebar" -ForegroundColor Yellow
Write-Host "Margins: Left=$LeftMargin Top=$TopMargin Right=$RightMargin Bottom=$BottomMargin" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure Kindle page is visible, then press ENTER" -ForegroundColor Yellow
Read-Host

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

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
    public static extern int GetSystemMetrics(int nIndex);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("kernel32.dll")]
    public static extern IntPtr GetConsoleWindow();
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@

[Win32]::SetProcessDPIAware() | Out-Null

# Get screen size
$screenWidth = [Win32]::GetSystemMetrics(0)
$screenHeight = [Win32]::GetSystemMetrics(1)

# Calculate capture region (exclude margins)
$captureX = $LeftMargin
$captureY = $TopMargin
$captureWidth = $screenWidth - $LeftMargin - $RightMargin
$captureHeight = $screenHeight - $TopMargin - $BottomMargin

Write-Host "Screen: ${screenWidth}x${screenHeight}" -ForegroundColor Cyan
Write-Host "Capture region: ${captureWidth}x${captureHeight} at ($captureX, $captureY)" -ForegroundColor Cyan
Write-Host ""

# Find Chrome
$chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -like "*Kindle*"} | Select-Object -First 1
if (-not $chrome) {
    $chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome" -and $_.MainWindowTitle -ne ""} | Select-Object -First 1
}

Write-Host "Starting in 3 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Hide PowerShell
$console = [Win32]::GetConsoleWindow()
[Win32]::ShowWindow($console, 6) | Out-Null
Start-Sleep -Milliseconds 500

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    # Focus Chrome if found
    if ($chrome) {
        [Win32]::SetForegroundWindow($chrome.MainWindowHandle) | Out-Null
        Start-Sleep -Milliseconds 300
    }
    
    # Capture content area only
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    $screenshot = New-Object System.Drawing.Bitmap $captureWidth, $captureHeight
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($captureX, $captureY, 0, 0, [System.Drawing.Size]::new($captureWidth, $captureHeight))
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    # Next page
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

# Restore PowerShell
[Win32]::ShowWindow($console, 9) | Out-Null

Write-Host ""
Write-Host "Complete! Saved to: $OutputDir" -ForegroundColor Green
