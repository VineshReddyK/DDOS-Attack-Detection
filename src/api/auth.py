"""
JWT authentication — issue and verify bearer tokens.
Set API_SECRET_KEY in environment (or .env) before production use.
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("API_SECRET_KEY", "change-me-in-production-use-a-strong-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))

_bearer = HTTPBearer(auto_error=True)


def _jwt():
    try:
        import jwt
        return jwt
    except ImportError:
        raise RuntimeError("PyJWT not installed. Add it to requirements.txt.")


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    jwt = _jwt()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    jwt = _jwt()
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        subject: str = payload.get("sub")
        if not subject:
            raise ValueError("Missing subject")
        return subject
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
