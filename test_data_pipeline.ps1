# test_data_pipeline.ps1 - Flexible CSV Testing
param(
    [string]$CsvPath = ""
)

# Auto-detect CSV file location
if (-not $CsvPath) {
    $possiblePaths = @(
        "diabetic_data.csv",
        ".\diabetic_data.csv", 
        "$env:USERPROFILE\Downloads\diabetic_data.csv",
        "$env:USERPROFILE\Desktop\diabetic_data.csv"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $CsvPath = $path
            Write-Host "✅ Found CSV at: $CsvPath" -ForegroundColor Green
            break
        }
    }
    
    if (-not $CsvPath) {
        Write-Host "❌ CSV file not found. Please specify path with -CsvPath parameter" -ForegroundColor Red
        Write-Host "Or place diabetic_data.csv in one of these locations:" -ForegroundColor Yellow
        $possiblePaths | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        exit 1
    }
}

Write-Host "🔍 Testing data pipeline with: $CsvPath" -ForegroundColor Cyan

# Test 1: Small dataset
Write-Host "
📊 Test 1: Small dataset processing..." -ForegroundColor Yellow
try {
    $lines = Get-Content $CsvPath -TotalCount 1001  # Header + 1000 rows
    $lines | Out-File test_data_small.csv -Encoding utf8
    
    poetry run python -c "
from ml.preprocess import preprocess_diabetes_data
from pathlib import Path
df, features = preprocess_diabetes_data(Path('test_data_small.csv'))
print(f'✅ Small dataset test: {len(df)} samples processed')
"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Small dataset test passed" -ForegroundColor Green
    } else {
        Write-Host "❌ Small dataset test failed" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Small dataset test error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Missing columns robustness
Write-Host "
📊 Test 2: Missing columns robustness..." -ForegroundColor Yellow
try {
    poetry run python -c "
import pandas as pd
df = pd.read_csv('$CsvPath')
# Remove some columns to test robustness  
df_missing = df.drop(columns=['weight', 'payer_code'], errors='ignore')
df_missing.to_csv('test_data_missing.csv', index=False)
print('Created test dataset with missing columns')

from ml.preprocess import preprocess_diabetes_data
from pathlib import Path
df, features = preprocess_diabetes_data(Path('test_data_missing.csv'))
print(f'✅ Missing columns test: {len(df)} samples processed')
"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Missing columns test passed" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing columns test failed" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Missing columns test error: $($_.Exception.Message)" -ForegroundColor Red
}

# Cleanup
Remove-Item test_data_small.csv -ErrorAction SilentlyContinue
Remove-Item test_data_missing.csv -ErrorAction SilentlyContinue

Write-Host "
✅ Data pipeline testing complete" -ForegroundColor Green
