import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ..config import settings
from ..models import McapDataRequest, OWAFile
from ..services import services
from ..utils import AppError, FileNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter()


# File Management Routes
@router.get("/api/list_files", response_model=List[OWAFile], tags=["files"])
async def list_files(repo_id: str) -> List[OWAFile]:
    """List all available MCAP+MKV files in a repository"""
    try:
        files = services.file_service.list_files(repo_id)
        logger.info(f"Fetched {len(files)} files for repo_id: {repo_id}")
        return files
    except AppError as e:
        raise e
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@router.get("/files/{file_path:path}", tags=["export"], responses={404: {"description": "Not found"}})
async def export_file(file_path: str):
    """Serve an MKV or MCAP file"""
    try:
        full_file_path = services.file_service.get_local_file_path(file_path)
        logger.info(f"Serving file: {full_file_path}")
        MEDIA_TYPE_MAP = {
            ".mkv": "video/matroska",
            ".mcap": "application/octet-stream",
        }

        media_type = MEDIA_TYPE_MAP.get(full_file_path.suffix, "application/octet-stream")
        return FileResponse(full_file_path.as_posix(), media_type=media_type)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


@router.post("/upload", tags=["import"], responses={404: {"description": "Not found"}})
async def import_file(mcap_file: UploadFile, mkv_file: UploadFile) -> OWAFile:
    """Import MCAP and MKV file pair"""
    # Check file size limits in public hosting mode
    if settings.PUBLIC_HOSTING_MODE:
        size_limit = 100 * 1024 * 1024  # 100MB
        if mcap_file.size > size_limit:
            raise HTTPException(
                status_code=400,
                detail="MCAP File size exceeds 100MB limit. Please self-host the viewer for files larger than 100MB. For more info, see https://open-world-agents.github.io/open-world-agents/data/viewer.",
            )
        if mkv_file.size > size_limit:
            raise HTTPException(
                status_code=400,
                detail="MKV File size exceeds 100MB limit. Please self-host the viewer for files larger than 100MB. For more info, see https://open-world-agents.github.io/open-world-agents/data/viewer.",
            )

    # Validate file extensions
    if not mcap_file.filename.endswith(".mcap"):
        raise HTTPException(status_code=400, detail="MCAP file must have .mcap extension")
    if not mkv_file.filename.endswith(".mkv"):
        raise HTTPException(status_code=400, detail="MKV file must have .mkv extension")

    # Make sure the base filenames match (excluding extensions)
    mcap_basename = Path(mcap_file.filename).stem
    mkv_basename = Path(mkv_file.filename).stem
    if mcap_basename != mkv_basename:
        raise HTTPException(status_code=400, detail="MCAP and MKV files must have the same base filename")

    # Ensure export path exists
    export_path = Path(settings.EXPORT_PATH)
    export_path.mkdir(exist_ok=True, parents=True)

    # Generate a random filename to avoid conflicts
    random_id = str(uuid.uuid4())
    random_basename = f"{mcap_basename}_{random_id}"
    random_mcap_filename = f"{random_basename}.mcap"
    random_mkv_filename = f"{random_basename}.mkv"

    # Save paths
    mcap_save_path = export_path / random_mcap_filename
    mkv_save_path = export_path / random_mkv_filename

    try:
        # Save files
        with mcap_save_path.open("wb") as f:
            shutil.copyfileobj(mcap_file.file, f)
        with mkv_save_path.open("wb") as f:
            shutil.copyfileobj(mkv_file.file, f)

        logger.info(
            f"Successfully imported files: {mcap_file.filename} as {random_mcap_filename}, {mkv_file.filename} as {random_mkv_filename}"
        )

        # Clear the cache for local repository to refresh the file list
        services.cache_service.file_list_cache.delete("local")

        return OWAFile(
            basename=random_basename,
            original_basename=mcap_basename,
            url=f"{random_basename}",
            size=mcap_file.size,
            local=True,
            url_mcap=f"{random_mcap_filename}",
            url_mkv=f"{random_mkv_filename}",
        )
    except Exception as e:
        logger.error(f"Error importing files: {str(e)}")
        # Clean up any partially uploaded files
        if mcap_save_path.exists():
            mcap_save_path.unlink()
        if mkv_save_path.exists():
            mkv_save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")


# MCAP Data Routes
@router.get("/api/mcap_info", tags=["mcap_data"])
async def get_mcap_info(mcap_filename: str, local: bool = True):
    """Return the `owl mcap info` command output"""
    try:
        return services.mcap_service.get_mcap_info(mcap_filename, local)
    except AppError as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting MCAP info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP info: {str(e)}")


@router.get("/api/mcap_metadata", tags=["mcap_data"])
async def get_mcap_metadata(mcap_filename: str, local: bool = True):
    """Get metadata about an MCAP file including time range and topics"""
    try:
        metadata = services.mcap_service.get_mcap_metadata(mcap_filename, local)
        return {"start_time": metadata.start_time, "end_time": metadata.end_time, "topics": list(metadata.topics)}
    except AppError as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting MCAP metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP metadata: {str(e)}")


@router.get("/api/mcap_data", tags=["mcap_data"])
async def get_mcap_data(
    mcap_filename: str,
    local: bool = True,
    start_time: Optional[int] = Query(None),
    end_time: Optional[int] = Query(None),
    window_size: Optional[int] = Query(10_000_000_000),  # Default 10-second window in nanoseconds
):
    """Get MCAP data for a specific time range"""
    try:
        request = McapDataRequest(
            mcap_filename=mcap_filename, local=local, start_time=start_time, end_time=end_time, window_size=window_size
        )
        return services.mcap_service.get_mcap_data(request)
    except AppError as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting MCAP data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP data: {str(e)}")
