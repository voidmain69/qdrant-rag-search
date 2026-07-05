from fastapi import APIRouter, Depends

from app.api.v1 import imports, products, search
from app.core.security import require_api_key

api_v1 = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
api_v1.include_router(products.router)
api_v1.include_router(search.router)
api_v1.include_router(imports.router)
