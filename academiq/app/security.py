"""Security primitives: admin API-key auth and per-IP rate limiting."""

from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

limiter = Limiter(key_func=get_remote_address)

_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(api_key: str | None = Depends(_admin_key_header)) -> str:
    settings = get_settings()
    if not api_key or api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key
