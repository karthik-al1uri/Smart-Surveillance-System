"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import create_access_token, get_current_user_payload
from src.api.dependencies import get_user_repo
from src.api.repositories import UserRepository
from src.api.schemas.auth_schemas import LoginRequest, TokenResponse, UserResponse

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, repo: UserRepository = Depends(get_user_repo)):
    """Authenticate and return a JWT access token."""
    user = repo.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password")
    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(access_token=token)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: dict = Depends(get_current_user_payload)):
    """Issue a fresh token for the current authenticated user."""
    token = create_access_token({"sub": payload["sub"], "role": payload.get("role", "viewer")})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(payload: dict = Depends(get_current_user_payload),
       repo: UserRepository = Depends(get_user_repo)):
    """Return the current authenticated user's profile."""
    user = repo.get_user(payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
