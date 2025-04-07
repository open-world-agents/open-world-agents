import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rich.logging import RichHandler

from mcap_owa.highlevel import OWAMcapReader
from owa_viewer.routers import export_file
from owa_viewer.schema import OWAFile
from owa_viewer.services import file_services
from owa_viewer.services.file_services import safe_join

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

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Cache for MCAP metadata
mcap_metadata_cache = {}

data_cache = {}


class McapMetadata:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.topics = set()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "featured_datasets": ["open-world-agents/example_dataset"]}
    )


@app.get("/viewer")
async def read_viewer(repo_id: str, request: Request):
    if data_cache.get(repo_id) is None:
        data_cache[repo_id] = file_services.list_file(repo_id)
    files = data_cache[repo_id]

    # TODO: uncomment below lines
    # size = sum(f.size for f in files)
    # # convert size to human readable format
    # size_units = ["iB", "KiB", "MiB", "GiB", "TiB"]
    # size_unit = min(len(size_units), np.floor(np.log2(size) / 10))
    # size /= 1024**size_unit
    # size = f"{size:.2f} {size_units[int(size_unit)]}"
    size = -1

    return templates.TemplateResponse(
        "viewer.html",
        {
            "request": request,
            "dataset_info": {"repo_id": repo_id, "files": len(files), "size": size},
        },
    )


@app.get("/api/list_files", response_model=list[OWAFile])
async def list_files(repo_id: str) -> list[OWAFile]:
    """List all available MCAP+MKV files"""

    return file_services.list_file(repo_id)


@app.get("/api/mcap_info")
async def get_mcap_info(mcap_filename: str, local: bool = True):
    """Return the `owl mcap info` command output"""
    try:
        if local:
            # join path
            mcap_path = safe_join(file_services.EXPORT_PATH, mcap_filename)

            if mcap_path is None or not mcap_path.exists():
                raise HTTPException(status_code=404, detail="MCAP file not found")

        else:
            # download file
            with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as temp_mcap:
                logger.info(f"Downloading MCAP file to: {temp_mcap.name}")

                resp = requests.get(mcap_filename)
                temp_mcap.write(resp.content)

                mcap_path = temp_mcap.name

        logger.info(f"Getting MCAP info for: {mcap_path}")

        try:
            # Run the `owl mcap info` command
            output = subprocess.check_output(["owl", "mcap", "info", str(mcap_path)], text=True)
            return {"info": output, "local": local}

        except Exception as e:
            logger.error(f"Error getting MCAP info: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting MCAP info: {str(e)}")

    finally:
        # remove
        if not local:
            Path(mcap_path).unlink(missing_ok=False)


@app.get("/api/mcap_metadata")
async def get_mcap_metadata(mcap_filename: str, local: bool = True):
    """Get metadata about an MCAP file including time range and topics"""
    mcap_path = None
    try:
        if mcap_filename not in mcap_metadata_cache:  # metadata not cached
            if local:
                # check if local file exists
                mcap_path = safe_join(file_services.EXPORT_PATH, mcap_filename)

                if mcap_path is None or not mcap_path.exists():
                    raise HTTPException(status_code=404, detail="MCAP file not found")

            else:
                # download remote file
                with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as temp_mcap:
                    logger.info(f"Downloading MCAP file to: {temp_mcap.name}")

                    resp = requests.get(mcap_filename)
                    temp_mcap.write(resp.content)

                mcap_path = temp_mcap.name

            # build metadata
            await build_mcap_metadata(mcap_path, mcap_filename)
        else:
            logger.info(f"Using cached metadata for {mcap_filename}")

        metadata = mcap_metadata_cache.get(mcap_filename)
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to create MCAP metadata")

        # Return metadata about the file
        return {"start_time": metadata.start_time, "end_time": metadata.end_time, "topics": list(metadata.topics)}
    finally:
        if not local and mcap_path:
            Path(mcap_path).unlink(missing_ok=False)


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
    try:
        # we need mcap data regardless of metadata
        if local:
            # check if local file exists
            mcap_path = safe_join(file_services.EXPORT_PATH, mcap_filename)

            if mcap_path is None or not mcap_path.exists():
                raise HTTPException(status_code=404, detail="MCAP file not found")

        else:
            # download remote file
            with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as temp_mcap:
                logger.info(f"Downloading MCAP file to: {temp_mcap.name}")

                resp = requests.get(mcap_filename)
                temp_mcap.write(resp.content)

            mcap_path = temp_mcap.name

        if mcap_filename not in mcap_metadata_cache:  # metadata not cached
            # build metadata
            await build_mcap_metadata(mcap_path, mcap_filename)
        else:
            logger.info(f"Using cached metadata for {mcap_filename}")

        metadata = mcap_metadata_cache.get(mcap_filename)
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
        if not local:
            Path(mcap_path).unlink(missing_ok=False)


async def build_mcap_metadata(mcap_path: Path, mcap_filename: str):
    """Build metadata about an MCAP file (time range, topics, etc.)"""

    if not Path(mcap_path).exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")

    logger.info(f"Building metadata for MCAP file: {mcap_path}")

    metadata = McapMetadata()

    try:
        with OWAMcapReader(mcap_path) as reader:
            metadata.start_time = reader.start_time
            metadata.end_time = reader.end_time
            metadata.topics = set(reader.topics)

            logger.info(
                f"Metadata built for {mcap_path}: {len(metadata.topics)} topics, "
                f"time range {metadata.start_time} to {metadata.end_time}"
            )

            # Store in the cache
            mcap_metadata_cache[mcap_filename] = metadata

            logger.info(f"Metadata cached for {mcap_filename}")

    except Exception as e:
        logger.error(f"Error building MCAP metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error building MCAP metadata: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    # huggingface space use 7860 port
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
