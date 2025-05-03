import logging
from typing import List

from fastapi import APIRouter, HTTPException

from ..models.file import OWAFile
from ..services.file_service import file_service
from ..utils.exceptions import AppError

router = APIRouter(tags=["files"])
logger = logging.getLogger(__name__)


@router.get("/api/list_files", response_model=List[OWAFile])
async def list_files(repo_id: str) -> List[OWAFile]:
    """
    List all available MCAP+MKV files in a repository

    Args:
        repo_id: Repository ID ('local' or Hugging Face dataset ID)

    Returns:
        List of OWAFile objects
    """
    try:
        files = file_service.list_files(repo_id)
        logger.info(f"Fetched {len(files)} files for repo_id: {repo_id}")
        return files
    except AppError as e:
        # Re-raise application errors
        raise e
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
