# Capture specific screen region - manual selection on first page
param(
    [int]$MaxPages = 500,
    [int]$WaitTime = 2,
    [string]$OutputDir = "$env:USERPROFILE\Documents\kindle_screenshots_region"
)

Write-Host "Region-Based Screenshot Capture" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Setup:" -ForegroundColor Yellow
Write-Host "1. Open Kindle to the first page you want to capture"
Write-Host "2. Press ENTER - a selection window will appear"
Write-Host "3. Click and drag to select the exact Kindle content area"
Write-Host "4. The script will capture that same region for all pages"
Write-Host ""

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Read-Host "Press ENTER to select capture region"

# Load assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Create selection form
$form = New-Object System.Windows.Forms.Form
$form.FormBorderStyle = 'None'
$form.WindowState = 'Maximized'
$form.BackColor = [System.Drawing.Color]::Black
$form.Opacity = 0.3
$form.TopMost = $true
$form.Cursor = [System.Windows.Forms.Cursors]::Cross

$startPoint = $null
$endPoint = $null

$form.Add_MouseDown({
    param($sender, $e)
    $script:startPoint = $e.Location
})

$form.Add_MouseUp({
    param($sender, $e)
    $script:endPoint = $e.Location
    $form.Close()
})

Write-Host "Click and drag to select the region to capture..." -ForegroundColor Cyan
$form.ShowDialog() | Out-Null

if (-not $startPoint -or -not $endPoint) {
    Write-Host "No region selected. Exiting." -ForegroundColor Red
    exit
}

# Calculate capture region
$x = [Math]::Min($startPoint.X, $endPoint.X)
$y = [Math]::Min($startPoint.Y, $endPoint.Y)
$width = [Math]::Abs($endPoint.X - $startPoint.X)
$height = [Math]::Abs($endPoint.Y - $startPoint.Y)

Write-Host ""
Write-Host "Selected region:" -ForegroundColor Green
Write-Host "  Position: ($x, $y)" -ForegroundColor Gray
Write-Host "  Size: ${width}x${height}" -ForegroundColor Gray
Write-Host ""

Start-Sleep -Seconds 2

# Capture loop
for ($i = 1; $i -le $MaxPages; $i++) {
    Write-Host "Page $i / $MaxPages" -ForegroundColor Cyan
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = Join-Path $OutputDir "page_$($i.ToString('000'))_$timestamp.png"
    
    # Capture the selected region
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
Write-Host "Complete! $MaxPages pages saved" -ForegroundColor Green
Write-Host "Location: $OutputDir" -ForegroundColor Cyan
