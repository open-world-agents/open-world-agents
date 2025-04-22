import logging
import subprocess
from typing import Optional

import numpy as np  # Add numpy import
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rich.logging import RichHandler

from mcap_owa.highlevel import OWAMcapReader
from owa_viewer.routers import export_file, import_file
from owa_viewer.schema import McapMetadata, OWAFile
from owa_viewer.services.file_manager import MCAP_METADATA_CACHE, OWAFILE_CACHE, FileManager, EXPORT_PATH

# Set up logging, use rich handler
logging.basicConfig(
    level="INFO",  # Or "INFO", "WARNING", etc.
    handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_level=True)],
)
logger = logging.getLogger(__name__)


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(export_file.router)
app.include_router(import_file.router)  # Include the new router

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_root(request: Request):
    featured_datasets = ["local", "open-world-agents/example_dataset", "open-world-agents/example_dataset2"]

    if EXPORT_PATH is None:
        featured_datasets.remove("local")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "featured_datasets": featured_datasets,
        },
    )


@app.get("/viewer")
async def read_viewer(repo_id: str, request: Request):
    if OWAFILE_CACHE.get(repo_id) is None:
        OWAFILE_CACHE[repo_id] = FileManager.list_files(repo_id)
    files = OWAFILE_CACHE[repo_id]

    # Calculate total size and format it as human-readable
    size = sum(f.size for f in files) if files else 0
    # convert size to human readable format
    size_units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size_unit = 0
    if size > 0:
        size_unit = min(len(size_units) - 1, int(np.floor(np.log2(max(1, size)) / 10)))
        size /= 1024**size_unit
    size_str = f"{size:.2f} {size_units[int(size_unit)]}"

    return templates.TemplateResponse(
        "viewer.html",
        {
            "request": request,
            "dataset_info": {"repo_id": repo_id, "files": len(files), "size": size_str},
        },
    )


@app.get("/api/list_files", response_model=list[OWAFile])
async def list_files(repo_id: str) -> list[OWAFile]:
    """List all available MCAP+MKV files"""

    return FileManager.list_files(repo_id)


@app.get("/api/mcap_info")
async def get_mcap_info(mcap_filename: str, local: bool = True):
    """Return the `owl mcap info` command output"""
    mcap_path = None
    is_temp = False

    try:
        mcap_path, is_temp = FileManager.get_mcap_path(mcap_filename, local)
        logger.info(f"Getting MCAP info for: {mcap_path}")

        try:
            # Run the `owl mcap info` command
            output = subprocess.check_output(["owl", "mcap", "info", str(mcap_path)], text=True)
            return {"info": output, "local": local}

        except Exception as e:
            logger.error(f"Error getting MCAP info: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting MCAP info: {str(e)}")

    finally:
        # Clean up temp file if needed
        if is_temp and mcap_path:
            FileManager.cleanup_temp_file(mcap_path)


@app.get("/api/mcap_metadata")
async def get_mcap_metadata(mcap_filename: str, local: bool = True):
    """Get metadata about an MCAP file including time range and topics"""
    mcap_path = None
    is_temp = False

    try:
        if mcap_filename not in MCAP_METADATA_CACHE:  # metadata not cached
            mcap_path, is_temp = FileManager.get_mcap_path(mcap_filename, local)

            FileManager.build_mcap_metadata(mcap_path, mcap_filename)
        else:
            logger.info(f"Using cached metadata for {mcap_filename}")

        metadata: McapMetadata = MCAP_METADATA_CACHE.get(mcap_filename)
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to create MCAP metadata")

        # Return metadata about the file
        return {"start_time": metadata.start_time, "end_time": metadata.end_time, "topics": list(metadata.topics)}
    finally:
        if is_temp and mcap_path:
            FileManager.cleanup_temp_file(mcap_path)


@app.get("/api/mcap_data")
async def get_mcap_data(
    mcap_filename: str,
    local: bool = True,
    start_time: Optional[int] = Query(None),
    end_time: Optional[int] = Query(None),
    window_size: Optional[int] = Query(10_000_000_000),  # Default 10-second window in nanoseconds
):
    """Get MCAP data for a specific time range"""
    mcap_path = None
    is_temp = False

    try:
        # We need the actual data
        mcap_path, is_temp = FileManager.get_mcap_path(mcap_filename, local)

        if mcap_filename not in MCAP_METADATA_CACHE:  # metadata not cached
            FileManager.build_mcap_metadata(mcap_path, mcap_filename)
        else:
            logger.info(f"Using cached metadata for {mcap_filename}")

        metadata = MCAP_METADATA_CACHE.get(mcap_filename)
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to create MCAP metadata")

        # If start_time is not provided, use the beginning of the file
        if start_time is None:
            start_time = metadata.start_time

        # If end_time is not provided, use start_time + window_size
        if end_time is None:
            end_time = start_time + window_size

        logger.info(f"Fetching MCAP data for time range: {start_time} to {end_time}")

        # Define topics we're interested in
        topics_of_interest = ["keyboard", "mouse", "screen", "window", "keyboard/state", "mouse/state"]

        # Initialize result structure
        topics_data = {topic: [] for topic in topics_of_interest}

        try:
            logger.info(f"Reading MCAP file: {mcap_path}")
            with OWAMcapReader(mcap_path) as reader:
                # Use the built-in filter parameters of iter_messages
                for topic, timestamp, message in reader.iter_decoded_messages(
                    topics=topics_of_interest, start_time=start_time, end_time=end_time
                ):
                    # topics_data[topic].append({"timestamp": timestamp, "data": message})
                    message["timestamp"] = timestamp
                    topics_data[topic].append(message)

            total_messages = sum(len(msgs) for msgs in topics_data.values())
            logger.info(f"Fetched {total_messages} messages for time range {start_time} to {end_time}")

            return topics_data

        except Exception as e:
            logger.error(f"Error fetching MCAP data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error fetching MCAP data: {str(e)}")

    finally:
        # Clean up temporary file if needed
        if is_temp and mcap_path:
            FileManager.cleanup_temp_file(mcap_path)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    # huggingface space use 7860 port
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
