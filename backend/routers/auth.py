from __future__ import annotations

import time
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
)
from backend.models.schemas import UserRegister, UserLogin, TokenResponse, RefreshRequest
from backend.models.db_models import User
from backend.core.config import get_settings
from sqlalchemy import select

router = APIRouter(prefix="/auth", tags=["auth"])

_settings = get_settings()
_login_attempts: dict[str, list[float]] = {}


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    window = 60.0

    # Periodic cleanup: evict stale entries to prevent unbounded growth
    _cleanup_attempts(now, window)

    if client_ip not in _login_attempts:
        _login_attempts[client_ip] = []
    _login_attempts[client_ip] = [t for t in _login_attempts[client_ip] if now - t < window]
    if len(_login_attempts[client_ip]) >= _settings.login_rate_limit:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")
    _login_attempts[client_ip].append(now)


_last_cleanup = 0.0


def _cleanup_attempts(now: float, window: float) -> None:
    global _last_cleanup
    if now - _last_cleanup < window:
        return
    _last_cleanup = now
    expired = [ip for ip, attempts in _login_attempts.items()
               if not any(now - t < window for t in attempts)]
    for ip in expired:
        del _login_attempts[ip]


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == body.username))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Username already taken")

    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    await db.flush()

    access = create_access_token(user.id, role=user.role)
    refresh = create_refresh_token(user.id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/login")
async def login(request: Request, body: UserLogin, db: AsyncSession = Depends(get_db)):
    _check_rate_limit(request.client.host if request.client else "unknown")

    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(user.id, role=user.role)
    refresh = create_refresh_token(user.id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh")
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Token is not a refresh token")

    user_id = int(payload["sub"])
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")

    access = create_access_token(user_id, role=payload.get("role", "user"))
    refresh_new = create_refresh_token(user_id)
    return TokenResponse(access_token=access, refresh_token=refresh_new)
