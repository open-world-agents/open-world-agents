import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query

from ..models.file import OWAFile
from ..services.file_service import file_service
from ..utils.exceptions import AppError

router = APIRouter(tags=["files"])
logger = logging.getLogger(__name__)


@router.get("/api/list_files", response_model=List[OWAFile])
async def list_files(
    repo_id: str,
    limit: int = Query(100, description="Maximum number of files to return", ge=1, le=1000),
    offset: int = Query(0, description="Number of files to skip", ge=0),
) -> List[OWAFile]:
    """
    List MCAP files in a repository with pagination

    Args:
        repo_id: Repository ID ('local' or Hugging Face dataset ID)
        limit: Maximum number of files to return (1-1000, default: 100)
        offset: Number of files to skip (default: 0)

    Returns:
        List of OWAFile objects
    """
    try:
        files = file_service.list_files(repo_id, limit=limit, offset=offset)
        logger.info(f"Fetched {len(files)} files for repo_id: {repo_id} (offset={offset}, limit={limit})")
        return files
    except AppError as e:
        # Re-raise application errors
        raise e
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
