import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ..config import settings
from ..models.file import OWAFile
from ..services.cache_service import cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["import"], responses={404: {"description": "Not found"}})


@router.post("")
async def import_file(
    mcap_file: UploadFile,
    mkv_file: UploadFile,
) -> OWAFile:
    """
    Import MCAP and MKV file pair

    Args:
        mcap_file: MCAP file to import
        mkv_file: MKV file to import

    Returns:
        OWAFile object with information about the imported files
    """
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

    # Create filenames with random ID
    random_mcap_filename = f"{random_basename}.mcap"
    random_mkv_filename = f"{random_basename}.mkv"

    # Save paths
    mcap_save_path = export_path / random_mcap_filename
    mkv_save_path = export_path / random_mkv_filename

    try:
        # Save MCAP file with random name
        with mcap_save_path.open("wb") as f:
            shutil.copyfileobj(mcap_file.file, f)

        # Save MKV file with random name
        with mkv_save_path.open("wb") as f:
            shutil.copyfileobj(mkv_file.file, f)

        logger.info(
            f"Successfully imported files: {mcap_file.filename} as {random_mcap_filename}, {mkv_file.filename} as {random_mkv_filename}"
        )

        # Clear the cache for local repository to refresh the file list
        cache_service.file_list_cache.delete("local")

        # Return success response with both original and random filenames
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
