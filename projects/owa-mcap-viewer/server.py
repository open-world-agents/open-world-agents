# server.py - improved with direct MCAP time filtering
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mcap.reader import make_reader
from pydantic import BaseModel

from mcap_owa.highlevel import OWAMcapReader

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

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Directory containing MCAP and MKV files
DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Cache for MCAP metadata
mcap_metadata_cache = {}


class FilePair(BaseModel):
    mcap_file: str
    mkv_file: str
    basename: str


class McapMetadata:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.topics = set()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/file_pairs", response_model=List[FilePair])
async def list_file_pairs():
    """List all available MCAP+MKV file pairs"""
    file_pairs = []
    data_path = Path(DATA_DIR)

    logger.info(f"Scanning directory: {data_path}")

    mcap_files = list(data_path.glob("*.mcap"))
    logger.info(f"Found {len(mcap_files)} MCAP files")

    for mcap_file in mcap_files:
        base_name = mcap_file.stem
        mkv_file = data_path / f"{base_name}.mkv"

        if mkv_file.exists():
            logger.info(f"Found matching pair: {mcap_file.name} and {mkv_file.name}")
            file_pairs.append(FilePair(mcap_file=str(mcap_file.name), mkv_file=str(mkv_file.name), basename=base_name))

    return file_pairs


@app.get("/api/mcap_metadata/{mcap_filename}")
async def get_mcap_metadata(mcap_filename: str):
    """Get metadata about an MCAP file including time range and topics"""
    mcap_path = Path(DATA_DIR) / mcap_filename

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
    mcap_path = Path(DATA_DIR) / mcap_filename

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


@app.get("/video/{video_filename}")
async def get_video(video_filename: str):
    """Serve an MKV video file"""
    video_path = Path(DATA_DIR) / video_filename

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise HTTPException(status_code=404, detail="Video file not found")

    logger.info(f"Serving video file: {video_path}")

    # BUG: below line does not support seeking. Use FileResponse instead
    # # Use StreamingResponse for better seeking support in large video files
    # def iterfile():
    #     with open(video_path, "rb") as f:
    #         yield from f

    # return StreamingResponse(iterfile(), media_type="video/x-matroska", headers={"Accept-Ranges": "bytes"})

    return FileResponse(str(video_path), media_type="video/x-matroska")


async def build_mcap_metadata(mcap_filename: str):
    """Build metadata about an MCAP file (time range, topics, etc.)"""
    mcap_path = Path(DATA_DIR) / mcap_filename

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
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
