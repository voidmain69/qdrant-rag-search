from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    if not getattr(request.app.state, "ready", False):
        response.status_code = 503
        return {"status": "starting"}
    return {
        "status": "ready",
        "indexed_code_points": len(request.app.state.code_index),
    }
