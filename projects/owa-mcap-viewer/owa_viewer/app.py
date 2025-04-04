import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mcap_owa.highlevel import OWAMcapReader
from owa_viewer.routers import export_file
from owa_viewer.schema import FilePair
from owa_viewer.services import file_services

# Set up logging
logging.basicConfig(level=logging.INFO)
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

# TODO: support huggingface & local dataset

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
        data_cache[repo_id] = file_services.list_filepair(repo_id)
    files = data_cache[repo_id]

    # TODO: deprecate list_filepair and use list_file instead and uncomment below lines
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


@app.get("/api/list_files", response_model=List[FilePair])
async def list_files(repo_id: str):
    """List all available MCAP+MKV file pairs"""

    return file_services.list_filepair(repo_id)


@app.get("/api/mcap_info/{mcap_filename}")
async def get_mcap_info(mcap_filename: str):
    """Return the `owl mcap info` command output"""
    mcap_path = Path(file_services.EXPORT_PATH) / mcap_filename

    if not mcap_path.exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")

    logger.info(f"Getting MCAP info for: {mcap_path}")

    try:
        # Run the `owl mcap info` command
        output = subprocess.check_output(["owl", "mcap", "info", str(mcap_path)], text=True)
        return {"info": output}

    except Exception as e:
        logger.error(f"Error getting MCAP info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting MCAP info: {str(e)}")


@app.get("/api/mcap_metadata/{mcap_filename}")
async def get_mcap_metadata(mcap_filename: str):
    """Get metadata about an MCAP file including time range and topics"""
    mcap_path = Path(file_services.EXPORT_PATH) / mcap_filename

    if not mcap_path.exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")

    # Get or build metadata about this file
    if mcap_filename not in mcap_metadata_cache:
        await build_mcap_metadata(mcap_filename)

    metadata = mcap_metadata_cache.get(mcap_filename)
    if not metadata:
        raise HTTPException(status_code=500, detail="Failed to create MCAP metadata")

    # Return metadata about the file
    return {"start_time": metadata.start_time, "end_time": metadata.end_time, "topics": list(metadata.topics)}


@app.get("/api/mcap_data/{mcap_filename}")
async def get_mcap_data(
    mcap_filename: str,
    start_time: Optional[int] = Query(None),
    end_time: Optional[int] = Query(None),
    window_size: Optional[int] = Query(10_000_000_000),  # Default 10-second window in nanoseconds
):
    """Get MCAP data for a specific time range"""
    mcap_path = Path(file_services.EXPORT_PATH) / mcap_filename

    if not mcap_path.exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")

    # Ensure we have metadata
    if mcap_filename not in mcap_metadata_cache:
        await build_mcap_metadata(mcap_filename)

    metadata = mcap_metadata_cache.get(mcap_filename)

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


async def build_mcap_metadata(mcap_filename: str):
    """Build metadata about an MCAP file (time range, topics, etc.)"""
    mcap_path = Path(file_services.EXPORT_PATH) / mcap_filename

    if not mcap_path.exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")

    logger.info(f"Building metadata for MCAP file: {mcap_path}")

    metadata = McapMetadata()

    try:
        with OWAMcapReader(mcap_path) as reader:
            metadata.start_time = reader.start_time
            metadata.end_time = reader.end_time
            metadata.topics = set(reader.topics)

            logger.info(
                f"Metadata built for {mcap_filename}: {len(metadata.topics)} topics, "
                f"time range {metadata.start_time} to {metadata.end_time}"
            )

            # Store in the cache
            mcap_metadata_cache[mcap_filename] = metadata

    except Exception as e:
        logger.error(f"Error building MCAP metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error building MCAP metadata: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    # huggingface space use 7860 port
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
