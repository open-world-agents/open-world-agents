import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.file_service import file_service
from ..utils.exceptions import FileNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["export"], responses={404: {"description": "Not found"}})


@router.get("/{file_path:path}")
async def export_file(file_path: str):
    """
    Serve an MKV or MCAP file

    Args:
        file_path: Path to the file to serve

    Returns:
        FileResponse with the requested file
    """
    try:
        # Use the file service to get the file path
        full_file_path = file_service.file_repository.get_local_file_path(file_path)

        logger.info(f"Serving file: {full_file_path}")

        # Determine media type based on file extension
        media_type = "video/x-matroska" if file_path.endswith(".mkv") else "application/octet-stream"

        return FileResponse(full_file_path.as_posix(), media_type=media_type)

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")
