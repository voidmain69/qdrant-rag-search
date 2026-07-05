import hmac

from fastapi import HTTPException, Request, status

from app.core.config import get_settings


async def require_api_key(request: Request) -> None:
    """Constant-time X-API-Key check. Empty API_KEYS disables auth (dev mode)."""
    keys = get_settings().api_key_list
    if not keys:
        return
    provided = request.headers.get("x-api-key", "")
    for key in keys:
        if hmac.compare_digest(provided.encode(), key.encode()):
            return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
