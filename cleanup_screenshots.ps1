# Clean up screenshot directories
param(
    [switch]$Confirm = $true
)

$dirs = @(
    "$env:USERPROFILE\Documents\kindle_screenshots",
    "$env:USERPROFILE\Documents\kindle_screenshots_simple",
    "$env:USERPROFILE\Documents\kindle_screenshots_content"
)

Write-Host "Screenshot Cleanup" -ForegroundColor Yellow
Write-Host "==================" -ForegroundColor Yellow
Write-Host ""

foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        $count = (Get-ChildItem $dir -Filter "*.png" | Measure-Object).Count
        $size = (Get-ChildItem $dir -Filter "*.png" | Measure-Object -Property Length -Sum).Sum / 1MB
        
        Write-Host "Found: $dir" -ForegroundColor Cyan
        Write-Host "  Files: $count screenshots" -ForegroundColor Gray
        Write-Host "  Size: $([math]::Round($size, 2)) MB" -ForegroundColor Gray
        
        if ($Confirm) {
            $response = Read-Host "  Delete all screenshots in this folder? (y/n)"
            if ($response -eq 'y') {
                Remove-Item "$dir\*.png" -Force
                Write-Host "  Deleted!" -ForegroundColor Green
            } else {
                Write-Host "  Skipped" -ForegroundColor Yellow
            }
        } else {
            Remove-Item "$dir\*.png" -Force
            Write-Host "  Deleted!" -ForegroundColor Green
        }
        Write-Host ""
    }
}

Write-Host "Cleanup complete!" -ForegroundColor Green
