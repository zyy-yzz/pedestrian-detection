from __future__ import annotations

from fastapi import Header, HTTPException

from backend.core.security import decode_access_token


async def get_current_user(authorization: str = Header(...)) -> dict:
    """Extract current user from Bearer token. Shared across all routers."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    try:
        payload = decode_access_token(authorization[7:])
        return {"user_id": int(payload["sub"]), "role": payload.get("role", "user")}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")