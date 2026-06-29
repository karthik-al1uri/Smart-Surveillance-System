"""
JWT-based authentication for the API.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.common.logger import get_logger

logger = get_logger("api.auth")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_AUTH_CFG: dict = {}


def configure_auth(config: dict) -> None:
    """Store auth config for later use by token functions.

    Args:
        config: Full project config dict; reads ``auth`` section.
    """
    _AUTH_CFG.update(config.get("auth", {}))


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt.

    Args:
        password: Plain-text password.

    Returns:
        Bcrypt hash string.
    """
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash.

    Args:
        plain: Plain-text candidate.
        hashed: Stored bcrypt hash.

    Returns:
        ``True`` if they match.
    """
    return _pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token.

    Args:
        data: Claims to encode (must include ``"sub"``).
        expires_delta: Optional custom expiry; defaults to config value.

    Returns:
        Encoded JWT string.
    """
    secret = _AUTH_CFG.get("secret_key", "change-this-in-production-please")
    algorithm = _AUTH_CFG.get("algorithm", "HS256")
    default_minutes = int(_AUTH_CFG.get("access_token_expire_minutes", 480))

    payload = dict(data)
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=default_minutes)
    )
    payload["exp"] = expire
    return jwt.encode(payload, secret, algorithm=algorithm)


def verify_token(token: str) -> dict:
    """Decode and verify a JWT token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        :class:`jose.JWTError`: If the token is invalid or expired.
    """
    secret = _AUTH_CFG.get("secret_key", "change-this-in-production-please")
    algorithm = _AUTH_CFG.get("algorithm", "HS256")
    return jwt.decode(token, secret, algorithms=[algorithm])


def get_current_user_payload(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency: decode JWT and return payload.

    Args:
        token: Bearer token from ``Authorization`` header.

    Returns:
        Decoded JWT payload.

    Raises:
        :class:`fastapi.HTTPException`: 401 if token is invalid.
    """
    try:
        payload = verify_token(token)
        username: str = payload.get("sub", "")
        if not username:
            raise JWTError("missing sub")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_role: str):
    """Return a FastAPI dependency that enforces a minimum role.

    Args:
        required_role: ``"admin"`` or ``"operator"``.

    Returns:
        FastAPI dependency callable.
    """
    def _check(payload: dict = Depends(get_current_user_payload)) -> dict:
        role = payload.get("role", "viewer")
        hierarchy = {"viewer": 0, "operator": 1, "admin": 2}
        if hierarchy.get(role, 0) < hierarchy.get(required_role, 99):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required.",
            )
        return payload
    return _check
