# Test capture - takes ONE screenshot and shows you
param(
    [string]$OutputDir = "$env:USERPROFILE\Documents\test_screenshot"
)

Write-Host "Test Screenshot" -ForegroundColor Green
Write-Host "===============" -ForegroundColor Green
Write-Host ""
Write-Host "This will take ONE screenshot of your current screen." -ForegroundColor Yellow
Write-Host "Make sure Kindle page is visible, then press ENTER" -ForegroundColor Yellow
Read-Host

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Write-Host "Taking screenshot in 2 seconds..." -ForegroundColor Cyan
Start-Sleep -Seconds 2

# Full screen capture
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$screenshot = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($screenshot)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)

$filename = Join-Path $OutputDir "test.png"
$screenshot.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$screenshot.Dispose()

Write-Host ""
Write-Host "Screenshot saved to:" -ForegroundColor Green
Write-Host "$filename" -ForegroundColor Cyan
Write-Host ""
Write-Host "Opening the screenshot..." -ForegroundColor Yellow

# Open the screenshot
Start-Process $filename

Write-Host ""
Write-Host "Check the screenshot:" -ForegroundColor Yellow
Write-Host "- Is the Kindle page fully visible?" -ForegroundColor Gray
Write-Host "- Is F11 fullscreen mode active?" -ForegroundColor Gray
Write-Host "- Can you see the complete page?" -ForegroundColor Gray
Write-Host ""
Write-Host "If it looks good, use capture_simple.ps1 with F11 mode" -ForegroundColor Green
