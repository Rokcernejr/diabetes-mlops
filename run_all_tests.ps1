# run_all_tests.ps1 - Comprehensive Testing Suite
Write-Host \"🧪 Running Comprehensive Test Suite\" -ForegroundColor Green

Write-Host \"
1️⃣ Basic Health Checks\" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 5
    Write-Host \"✅ API Health: $($health.status)\" -ForegroundColor Green
} catch {
    Write-Host \"❌ API Health Check Failed\" -ForegroundColor Red
    exit 1
}


Write-Host \"
2️⃣ Data Pipeline Testing\" -ForegroundColor Cyan  
.\test_data_pipeline.ps1

Write-Host \"
3️⃣ Model Consistency Testing\" -ForegroundColor Cyan
.\test_model_consistency.ps1


Write-Host \"
🎉 All Tests Complete!\" -ForegroundColor Green
