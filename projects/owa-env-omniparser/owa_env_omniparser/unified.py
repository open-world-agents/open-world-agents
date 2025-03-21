"""
Unified OmniParser interface.

This module provides a unified interface for OmniParser functionality,
supporting both embedded and API modes with automatic fallback.
"""

import os
import logging
import time
from typing import Dict, Any, Union
from io import BytesIO

from owa.registry import CALLABLES

logger = logging.getLogger(__name__)


@CALLABLES.register("screen.parse")
def parse_screen_unified(screen_image: Union[str, bytes, BytesIO]) -> Dict[str, Any]:
    """
    Parse a screen image using OmniParser in embedded or API mode.

    This function provides a unified interface that will use either the embedded
    OmniParser implementation or the API client based on configuration. It also
    supports automatic fallback if the primary mode fails.

    Args:
        screen_image: The screen image to parse.
            Can be a base64 encoded string, bytes, or BytesIO object.

    Returns:
        Dict containing parsed UI elements and visualization image

    Raises:
        RuntimeError: If both embedded and API modes fail
    """
    mode = os.getenv("OMNIPARSER_MODE", "embedded").lower()
    fallback_enabled = os.getenv("OMNIPARSER_FALLBACK_ENABLED", "true").lower() in ("true", "1", "yes")

    # Caching settings
    use_cache = os.getenv("OMNIPARSER_USE_CACHE", "false").lower() in ("true", "1", "yes")
    cache_ttl = int(os.getenv("OMNIPARSER_CACHE_TTL", "300"))  # 5 minutes default

    # Use cache if available and valid
    if use_cache and hasattr(parse_screen_unified, "_cache") and hasattr(parse_screen_unified, "_cache_time"):
        cache_age = time.time() - parse_screen_unified._cache_time
        if cache_age < cache_ttl:
            # For simple caching, we don't compare the actual image content
            # In a production implementation, you'd want to hash the image for comparison
            logger.info("Using cached OmniParser result")
            return parse_screen_unified._cache_result

    start_time = time.time()
    primary_error = None

    # Try primary mode
    try:
        if mode == "api":
            logger.info("Using API mode for OmniParser")
            result = CALLABLES["screen.parse_omniparser_api"](screen_image)
        else:
            logger.info("Using embedded mode for OmniParser")
            result = CALLABLES["screen.parse_omniparser"](screen_image)

        # Cache successful result
        if use_cache:
            parse_screen_unified._cache_result = result
            parse_screen_unified._cache_time = time.time()

        logger.info(f"OmniParser completed in {time.time() - start_time:.2f}s using {mode} mode")
        return result

    except Exception as e:
        primary_error = e
        logger.error(f"Primary mode ({mode}) failed: {str(e)}")

        # Bail out if fallback is disabled
        if not fallback_enabled:
            logger.warning("Fallback is disabled, raising the original error")
            raise

    # Try fallback mode
    try:
        fallback_mode = "embedded" if mode == "api" else "api"
        logger.info(f"Falling back to {fallback_mode} mode")

        if fallback_mode == "api":
            result = CALLABLES["screen.parse_omniparser_api"](screen_image)
        else:
            result = CALLABLES["screen.parse_omniparser"](screen_image)

        # Cache successful result
        if use_cache:
            parse_screen_unified._cache_result = result
            parse_screen_unified._cache_time = time.time()

        logger.info(f"OmniParser completed in {time.time() - start_time:.2f}s using {fallback_mode} mode (fallback)")
        return result

    except Exception as fallback_error:
        logger.error(f"Fallback mode ({fallback_mode}) also failed: {str(fallback_error)}")

        # Both modes failed, raise a combined error
        raise RuntimeError(
            f"OmniParser failed in both modes. Primary ({mode}): {str(primary_error)}. "
            f"Fallback ({fallback_mode}): {str(fallback_error)}"
        )


@CALLABLES.register("ui.find_and_click")
def find_and_click_element(element_description: str, similarity_threshold: float = 0.7, max_retries: int = 3) -> bool:
    """
    Find a UI element by description and click on it.

    This function combines OmniParser's element detection with mouse control.

    Args:
        element_description: Description of the element to find
        similarity_threshold: Minimum similarity score (0.0-1.0)
        max_retries: Maximum number of retry attempts

    Returns:
        True if element was found and clicked, False otherwise
    """
    # Check if required functions are available
    if "screen.capture" not in CALLABLES:
        logger.error("screen.capture function is not available")
        return False

    if "mouse.move_to" not in CALLABLES or "mouse.click" not in CALLABLES:
        logger.error("Mouse control functions are not available")
        return False

    # Try to find and click the element
    for attempt in range(max_retries):
        try:
            # Capture screen
            screen_image = CALLABLES["screen.capture"]()
            if not screen_image:
                logger.error("Failed to capture screen")
                continue

            # Parse screen with OmniParser
            parsed_elements = parse_screen_unified(screen_image)

            # Find element by description
            element = CALLABLES["screen.get_element_by_description"](
                parsed_elements, element_description, similarity_threshold
            )

            if element and "center_coordinates" in element:
                # Move mouse and click
                x, y = element["center_coordinates"]
                CALLABLES["mouse.move_to"](x, y)
                CALLABLES["mouse.click"]()

                logger.info(f"Clicked on element '{element_description}' at ({x}, {y})")
                return True
            else:
                logger.warning(f"Element '{element_description}' not found (attempt {attempt + 1}/{max_retries})")

        except Exception as e:
            logger.error(f"Error finding/clicking element: {str(e)}")

        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(1)

    logger.error(f"Failed to find and click element '{element_description}' after {max_retries} attempts")
    return False
