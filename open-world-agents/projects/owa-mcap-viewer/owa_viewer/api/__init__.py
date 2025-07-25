from fastapi import APIRouter

from .data_routes import router as data_router
from .export_routes import router as export_router
from .file_routes import router as file_router
from .import_routes import router as import_router

# Create a combined router
router = APIRouter()
router.include_router(file_router)
router.include_router(data_router)
router.include_router(export_router)
router.include_router(import_router)
