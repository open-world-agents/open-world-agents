import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..models.mcap import McapDataRequest
from ..services.mcap_service import mcap_service
from ..utils.exceptions import AppError

router = APIRouter(tags=["mcap_data"])
logger = logging.getLogger(__name__)


@router.get("/api/mcap_info")
async def get_mcap_info(mcap_filename: str, local: bool = True):
    """
    Return the `owl mcap info` command output

    Args:
        mcap_filename: Path or URL to the MCAP file
        local: Whether the file is local

    Returns:
        Dictionary with MCAP info and local flag
    """
    try:
        return mcap_service.get_mcap_info(mcap_filename, local)
    except AppError as e:
        # Re-raise application errors
        raise e
    except Exception as e:
        logger.error(f"Error getting MCAP info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP info: {str(e)}")


@router.get("/api/mcap_metadata")
async def get_mcap_metadata(mcap_filename: str, local: bool = True):
    """
    Get metadata about an MCAP file including time range and topics

    Args:
        mcap_filename: Path or URL to the MCAP file
        local: Whether the file is local

    Returns:
        Dictionary with start_time, end_time, and topics
    """
    try:
        metadata = mcap_service.get_mcap_metadata(mcap_filename, local)
        return {"start_time": metadata.start_time, "end_time": metadata.end_time, "topics": list(metadata.topics)}
    except AppError as e:
        # Re-raise application errors
        raise e
    except Exception as e:
        logger.error(f"Error getting MCAP metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP metadata: {str(e)}")


@router.get("/api/mcap_data")
async def get_mcap_data(
    mcap_filename: str,
    local: bool = True,
    start_time: Optional[int] = Query(None),
    end_time: Optional[int] = Query(None),
    window_size: Optional[int] = Query(10_000_000_000),  # Default 10-second window in nanoseconds
):
    """
    Get MCAP data for a specific time range

    Args:
        mcap_filename: Path or URL to the MCAP file
        local: Whether the file is local
        start_time: Start time in nanoseconds
        end_time: End time in nanoseconds
        window_size: Window size in nanoseconds

    Returns:
        Dictionary mapping topics to lists of messages
    """
    try:
        request = McapDataRequest(
            mcap_filename=mcap_filename, local=local, start_time=start_time, end_time=end_time, window_size=window_size
        )
        return mcap_service.get_mcap_data(request)
    except AppError as e:
        # Re-raise application errors
        raise e
    except Exception as e:
        logger.error(f"Error getting MCAP data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP data: {str(e)}")
