import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from mcap_owa.highlevel import OWAMcapReader

from ..models.mcap import McapDataRequest, McapMetadata
from ..services.cache_service import cache_service
from ..services.file_service import file_service
from ..utils.exceptions import FileNotFoundError, McapProcessingError

logger = logging.getLogger(__name__)


class McapService:
    """Service for handling MCAP file operations"""

    def get_mcap_info(self, mcap_filename: str, local: bool) -> Dict[str, Any]:
        """
        Get information about an MCAP file using the owl CLI

        Args:
            mcap_filename: Path or URL to the MCAP file
            local: Whether the file is local

        Returns:
            Dictionary with MCAP info and local flag
        """
        mcap_path, is_temp = None, False

        try:
            mcap_path, is_temp = file_service.get_file_path(mcap_filename, local)
            logger.info(f"Getting MCAP info for: {mcap_path}")

            try:
                # Run the `owl mcap info` command
                output = subprocess.check_output(["owl", "mcap", "info", str(mcap_path)], text=True)

                # Parse only the relevant part of the output
                output = output[output.find("library:") :]
                return {"info": output, "local": local}

            except Exception as e:
                logger.error(f"Error getting MCAP info: {e}", exc_info=True)
                raise McapProcessingError(f"Error getting MCAP info: {str(e)}")

        finally:
            if is_temp and mcap_path:
                file_service.cleanup_temp_file(mcap_path)

    def get_mcap_metadata(self, mcap_filename: str, local: bool) -> McapMetadata:
        """
        Get metadata for an MCAP file

        Args:
            mcap_filename: Path or URL to the MCAP file
            local: Whether the file is local

        Returns:
            McapMetadata object
        """
        # Check cache first
        cache_key = f"{'local' if local else 'remote'}:{mcap_filename}"
        cached_metadata = cache_service.get_metadata(cache_key)
        if cached_metadata:
            logger.info(f"Using cached metadata for {mcap_filename}")
            return McapMetadata(**cached_metadata)

        # Need to process the file
        mcap_path, is_temp = None, False

        try:
            mcap_path, is_temp = file_service.get_file_path(mcap_filename, local)
            logger.info(f"Building metadata for MCAP file: {mcap_path}")

            if not Path(mcap_path).exists():
                raise FileNotFoundError(mcap_filename)

            try:
                with OWAMcapReader(mcap_path) as reader:
                    metadata = McapMetadata(
                        start_time=reader.start_time,
                        end_time=reader.end_time,
                        topics=reader.topics,
                    )

                    # Cache the metadata
                    cache_service.set_metadata(cache_key, metadata.model_dump())

                    logger.info(
                        f"Metadata built for {mcap_path}: {len(metadata.topics)} topics, "
                        f"time range {metadata.start_time} to {metadata.end_time}"
                    )

                    return metadata

            except Exception as e:
                logger.error(f"Error building MCAP metadata: {e}", exc_info=True)
                raise McapProcessingError(f"Error building MCAP metadata: {str(e)}")

        finally:
            if is_temp and mcap_path:
                file_service.cleanup_temp_file(mcap_path)

    def get_mcap_data(self, request: McapDataRequest) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract data from MCAP file for specified time range and topics

        Args:
            request: McapDataRequest with parameters

        Returns:
            Dictionary mapping topics to lists of messages
        """
        mcap_path, is_temp = None, False

        try:
            # Get the file
            mcap_path, is_temp = file_service.get_file_path(request.mcap_filename, request.local)

            # Get metadata to determine time range if needed
            metadata = self.get_mcap_metadata(request.mcap_filename, request.local)

            # Set default time range if not provided
            start_time = request.start_time if request.start_time is not None else metadata.start_time
            end_time = request.end_time if request.end_time is not None else (start_time + request.window_size)

            logger.info(f"Fetching MCAP data for time range: {start_time} to {end_time}")

            # Define topics we're interested in
            topics_of_interest = ["keyboard", "mouse", "screen", "window", "keyboard/state", "mouse/state"]

            # Initialize result structure
            topics_data = {topic: [] for topic in topics_of_interest}

            try:
                logger.info(f"Reading MCAP file: {mcap_path}")
                with OWAMcapReader(mcap_path) as reader:
                    for mcap_msg in reader.iter_messages(
                        topics=topics_of_interest, start_time=start_time, end_time=end_time
                    ):
                        mcap_msg.decoded["timestamp"] = mcap_msg.timestamp
                        topics_data[mcap_msg.topic].append(mcap_msg.decoded)

                total_messages = sum(len(msgs) for msgs in topics_data.values())
                logger.info(f"Fetched {total_messages} messages for time range {start_time} to {end_time}")

                return topics_data

            except Exception as e:
                logger.error(f"Error fetching MCAP data: {e}", exc_info=True)
                raise McapProcessingError(f"Error fetching MCAP data: {str(e)}")

        finally:
            if is_temp and mcap_path:
                file_service.cleanup_temp_file(mcap_path)


# Create singleton instance
mcap_service = McapService()
