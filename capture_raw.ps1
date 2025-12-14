# Simple capture - full screen only
param(
    [int]$MaxPages = 560,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_raw"
)

Write-Host "Simple Full Screen Capture" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green
Write-Host ""
Write-Host "Press ENTER to start (Python will crop later)" -ForegroundColor Yellow
Read-Host

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Find Chrome
$chrome = Get-Process | Where-Object {$_.ProcessName -eq "chrome"} | Select-Object -First 1

Write-Host "Starting..." -ForegroundColor Green
Start-Sleep -Seconds 2

for ($i = 1; $i -le $MaxPages; $i++) {
    if ($i % 10 -eq 0) { Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan }
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $screenshot = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
    $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    
    $screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $screenshot.Dispose()
    
    [System.Windows.Forms.SendKeys]::SendWait("{RIGHT}")
    Start-Sleep -Seconds $WaitTime
}

Write-Host ""
Write-Host "Done! Now run Python script to crop" -ForegroundColor Green
