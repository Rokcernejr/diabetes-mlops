from cryptography.fernet import Fernet
import os

class EncryptionMiddleware:
    def __init__(self):
        key = os.getenv('ENCRYPTION_KEY')
        self.cipher = Fernet(key) if key else None
    
    def encrypt_pii(self, data):
        if self.cipher and self.contains_pii(data):
            return self.cipher.encrypt(data.encode())
        return data
        
    def contains_pii(self, data):
        # Check if data contains PII
        return any(field in str(data).lower() for field in ['name', 'ssn', 'dob'])