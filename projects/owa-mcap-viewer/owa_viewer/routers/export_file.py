import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.file_services import EXPORT_PATH

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/files", tags=["export"], responses={404: {"description": "Not found"}})


@router.get("/{filename}")
async def export_file(filename: str):
    """Serve an MKV video file"""
    file_path = Path(EXPORT_PATH) / filename

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"Serving file: {file_path}")

    # BUG: below line does not support seeking. Use FileResponse instead
    # # Use StreamingResponse for better seeking support in large video files
    # def iterfile():
    #     with open(video_path, "rb") as f:
    #         yield from f

    # return StreamingResponse(iterfile(), media_type="video/x-matroska", headers={"Accept-Ranges": "bytes"})

    return FileResponse(file_path.as_posix(), media_type="video/x-matroska")
