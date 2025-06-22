import os
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

# Configuration
JWKS_URL = os.getenv("JWKS_URL")
ISSUER = os.getenv("ISSUER")
AUDIENCE = "mlops-diabetes-api"

# Security scheme
security = HTTPBearer(auto_error=False)

# JWK client (only if JWKS_URL is configured)
_jwk_client = None
if JWKS_URL:
    _jwk_client = PyJWKClient(JWKS_URL)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify JWT token (optional in development)"""
    
    # Skip authentication in development
    if os.getenv("ENVIRONMENT") == "development":
        return {"sub": "dev-user", "permissions": ["read", "write"]}
    
    # Require authentication in production
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    if not _jwk_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )
    
    try:
        # Get signing key
        signing_key = _jwk_client.get_signing_key_from_jwt(credentials.credentials).key
        
        # Decode and verify token
        payload = jwt.decode(
            credentials.credentials,
            signing_key,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
        
        return payload
        
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
