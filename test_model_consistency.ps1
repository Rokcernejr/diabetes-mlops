# test_model_consistency.ps1
Write-Host \"🧠 Testing Model Consistency\" -ForegroundColor Green

poetry run python -c \"
from ml.consistency_test import test_model_consistency
from pathlib import Path

print('🔄 Running model consistency test...')
try:
    results = test_model_consistency(Path('data/processed.parquet'), n_trials=3)
    
    print(f'📊 Consistency Results:')
    print(f'   Trials: {results[\"trials\"]}')
    print(f'   AUC Scores: {[f\"{x:.4f}\" for x in results[\"auc_scores\"]]}')
    print(f'   Mean AUC: {results[\"mean_auc\"]:.4f}')
    print(f'   Std AUC: {results[\"std_auc\"]:.4f}')
    
    if results['is_consistent']:
        print('✅ Model training is consistent')
    else:
        print('⚠️ Model training shows high variance')
        
except Exception as e:
    print(f'❌ Consistency test failed: {e}')
\"
