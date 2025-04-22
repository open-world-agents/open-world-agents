import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ..schema import OWAFile
from ..services.file_manager import EXPORT_PATH, OWAFILE_CACHE, PUBLIC_HOSTING_MODE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["import"], responses={404: {"description": "Not found"}})


@router.post("")
async def import_file(
    mcap_file: UploadFile,
    mkv_file: UploadFile,
) -> OWAFile:
    """Import MCAP and MKV file pair"""

    # if public hosted and file size exceeds 100MB, raise an error
    if PUBLIC_HOSTING_MODE:
        if mcap_file.size > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="MCAP File size exceeds 100MB limit. Please self-host the viewer for files larger than 100MB. For more info, see https://open-world-agents.github.io/open-world-agents/data/viewer.",
            )
        if mkv_file.size > 100 * 1024 * 1024:
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

    # Ensure EXPORT_PATH exists
    export_path = Path(EXPORT_PATH)
    export_path.mkdir(exist_ok=True, parents=True)

    # Save the files
    mcap_save_path = export_path / mcap_file.filename
    mkv_save_path = export_path / mkv_file.filename

    # Check if files already exist
    if mcap_save_path.exists() or mkv_save_path.exists():
        raise HTTPException(status_code=409, detail="Files with these names already exist")

    try:
        # Save MCAP file
        with mcap_save_path.open("wb") as f:
            shutil.copyfileobj(mcap_file.file, f)

        # Save MKV file
        with mkv_save_path.open("wb") as f:
            shutil.copyfileobj(mkv_file.file, f)

        logger.info(f"Successfully imported files: {mcap_file.filename}, {mkv_file.filename}")

        # Force refresh of the file list cache
        if "local" in OWAFILE_CACHE:
            OWAFILE_CACHE.pop("local")  # Clear the cache for local repository

        # Return success response
        return OWAFile(
            basename=mcap_basename,
            url=f"{mcap_file.filename}",
            size=mcap_file.size,
            local=True,
            url_mcap=f"{mcap_file.filename}",
            url_mkv=f"{mkv_file.filename}",
        )

    except Exception as e:
        logger.error(f"Error importing files: {str(e)}")
        # Clean up any partially uploaded files
        if mcap_save_path.exists():
            mcap_save_path.unlink()
        if mkv_save_path.exists():
            mkv_save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")
