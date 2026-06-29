"""User management routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import require_role
from src.api.dependencies import get_user_repo
from src.api.repositories import UserRepository
from src.api.schemas.auth_schemas import UserCreate, UserResponse

router = APIRouter()


@router.get("", response_model=List[UserResponse])
def list_users(
    repo: UserRepository = Depends(get_user_repo),
    _: dict = Depends(require_role("admin")),
):
    """List all users (admin only)."""
    return repo.list_users()


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserCreate,
    repo: UserRepository = Depends(get_user_repo),
    _: dict = Depends(require_role("admin")),
):
    """Create a new user (admin only)."""
    if repo.get_user(body.username):
        raise HTTPException(status_code=409, detail="Username already exists")
    return repo.create_user(
        username=body.username,
        password=body.password,
        role=body.role,
        full_name=body.full_name,
    )


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    updates: dict,
    repo: UserRepository = Depends(get_user_repo),
    _: dict = Depends(require_role("admin")),
):
    """Update user fields (admin only)."""
    user = repo.update_user(user_id, updates)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/{user_id}")
def delete_user(
    user_id: str,
    repo: UserRepository = Depends(get_user_repo),
    _: dict = Depends(require_role("admin")),
):
    """Delete a user (admin only)."""
    if not repo.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": user_id}
