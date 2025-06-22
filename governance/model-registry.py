from typing import Dict, List
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def register_model(self, version: str, metrics: Dict, metadata: Dict):
        self.models[version] = {
            'metrics': metrics,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'
        }
    
    def promote_to_production(self, version: str, approver: str):
        if version in self.models:
            self.models[version]['status'] = 'production'
            self.models[version]['approved_by'] = approver
            self.models[version]['promoted_at'] = datetime.now().isoformat()
    
    def get_production_models(self) -> List[str]:
        return [v for v, data in self.models.items() if data['status'] == 'production']
