from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    state = request.app.state
    if not getattr(state, "ready", False):
        response.status_code = 503
        return {"status": "starting"}
    if not await state.qdrant.ping():
        response.status_code = 503
        return {"status": "degraded", "detail": "Qdrant is unreachable"}
    return {
        "status": "ready",
        "indexed_code_points": len(state.code_index),
    }
