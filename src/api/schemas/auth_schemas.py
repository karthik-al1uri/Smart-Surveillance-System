"""Auth request/response schemas."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "operator"
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    username: str
    full_name: Optional[str]
    role: str
    enabled: bool

    model_config = {"from_attributes": True}
