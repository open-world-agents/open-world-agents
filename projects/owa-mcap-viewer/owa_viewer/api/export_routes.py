import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..services.file_service import file_service
from ..services.media_service import media_service
from ..utils.exceptions import FileNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["export"], responses={404: {"description": "Not found"}})


@router.get("/media")
async def serve_media_reference(
    mcap_filename: str = Query(..., description="MCAP filename containing the media reference"),
    media_uri: str = Query(..., description="Media URI from MediaRef"),
    local: bool = Query(True, description="Whether the MCAP file is local"),
):
    """
    Serve media content from a MediaRef URI.

    Args:
        mcap_filename: Name of the MCAP file containing the reference
        media_uri: URI from MediaRef
        local: Whether the MCAP file is local

    Returns:
        Media content response
    """
    return media_service.serve_media_from_reference(mcap_filename, media_uri, local)


@router.get("/primary_video")
async def get_primary_video(
    mcap_filename: str = Query(..., description="MCAP filename"),
    local: bool = Query(True, description="Whether the MCAP file is local"),
):
    """
    Get the primary video reference for the video player.

    Args:
        mcap_filename: Name of the MCAP file
        local: Whether the MCAP file is local

    Returns:
        Primary video URI or error if no video found
    """
    try:
        logger.info(f"Getting primary video for MCAP: {mcap_filename} (local={local})")
        video_uri = media_service.get_primary_video_reference(mcap_filename, local)
        if video_uri:
            logger.info(f"Found primary video URI: {video_uri}")
            # Return the media content directly
            return media_service.serve_media_from_reference(mcap_filename, video_uri, local)
        else:
            logger.warning(f"No video reference found in MCAP file: {mcap_filename}")
            raise HTTPException(status_code=404, detail=f"No video reference found in MCAP file: {mcap_filename}")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting primary video for {mcap_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting primary video: {str(e)}")


@router.get("/validate_media")
async def validate_media_references(
    mcap_filename: str = Query(..., description="MCAP filename"),
    local: bool = Query(True, description="Whether the MCAP file is local"),
):
    """
    Validate all media references in an MCAP file.

    Args:
        mcap_filename: Name of the MCAP file
        local: Whether the MCAP file is local

    Returns:
        Validation results
    """
    try:
        validation_results = media_service.validate_media_references(mcap_filename, local)
        return validation_results
    except Exception as e:
        logger.error(f"Error validating media references for {mcap_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error validating media references: {str(e)}")


@router.get("/debug_screen_messages")
async def debug_screen_messages(
    mcap_filename: str = Query(..., description="MCAP filename"),
    local: bool = Query(True, description="Whether the MCAP file is local"),
    limit: int = Query(5, description="Number of messages to examine"),
):
    """
    Debug endpoint to examine screen messages in an MCAP file.
    """
    try:
        from mcap_owa.highlevel import OWAMcapReader

        mcap_path, is_temp = media_service.file_service.get_file_path(mcap_filename, local)

        messages_info = []
        with OWAMcapReader(str(mcap_path)) as reader:
            count = 0
            for mcap_msg in reader.iter_messages(topics=["screen"]):
                if count >= limit:
                    break
                count += 1

                screen_data = mcap_msg.decoded

                msg_info = {
                    "message_index": count,
                    "timestamp": mcap_msg.log_time,
                    "has_media_ref": hasattr(screen_data, "media_ref") and screen_data.media_ref is not None,
                    "has_path": hasattr(screen_data, "path") and screen_data.path is not None,
                }

                if hasattr(screen_data, "media_ref") and screen_data.media_ref:
                    media_ref = screen_data.media_ref
                    msg_info["media_ref"] = {
                        "uri": getattr(media_ref, "uri", "N/A"),
                        "is_video": getattr(media_ref, "is_video", False),
                        "pts_ns": getattr(media_ref, "pts_ns", None),
                        "is_embedded": getattr(media_ref, "is_embedded", False),
                    }

                if hasattr(screen_data, "path") and screen_data.path:
                    msg_info["legacy_path"] = screen_data.path
                    msg_info["legacy_pts"] = getattr(screen_data, "pts", None)

                messages_info.append(msg_info)

        # Clean up temporary MCAP file if needed
        if is_temp:
            media_service.file_service.cleanup_temp_file(mcap_path)

        return {"mcap_filename": mcap_filename, "messages_examined": len(messages_info), "messages": messages_info}

    except Exception as e:
        logger.error(f"Error debugging screen messages for {mcap_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error debugging screen messages: {str(e)}")


@router.get("/{file_path:path}")
async def export_file(file_path: str):
    """
    Serve an MKV or MCAP file (legacy endpoint for backward compatibility)

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
