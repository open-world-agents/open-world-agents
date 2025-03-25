"""
Embedded OmniParser functionality for Open World Agents.

This module provides callable functions for parsing UI screenshots using
OmniParser in an embedded mode.
"""

import os
import base64
import logging
import importlib.util
from typing import Dict, Any, Union, Optional
from io import BytesIO
from pathlib import Path

from owa.registry import CALLABLES
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

# Singleton OmniParser instance
_omniparser_instance = None


def _import_omniparser():
    """
    Dynamically import OmniParser from the available source.

    Returns:
        OmniParser class if available, otherwise None

    Raises:
        ImportError: If OmniParser dependencies are not installed
    """
    try:
        # Try to import directly first (if OmniParser is installed)
        from OmniParser.util.omniparser import OmniParser

        return OmniParser
    except ImportError:
        # Check if OmniParser is available in the workspace
        omniparser_path = os.getenv("OMNIPARSER_SOURCE_PATH")
        if omniparser_path:
            try:
                # Add OmniParser directory to path
                omniparser_path = Path(omniparser_path).expanduser().resolve()
                if not omniparser_path.exists():
                    logger.error(f"OmniParser source path does not exist: {omniparser_path}")
                    return None

                # Import using path
                spec = importlib.util.spec_from_file_location(
                    "omniparser", str(omniparser_path / "util" / "omniparser.py")
                )
                if spec and spec.loader:
                    omniparser_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(omniparser_module)
                    return omniparser_module.OmniParser
            except Exception as e:
                logger.error(f"Failed to import OmniParser from {omniparser_path}: {str(e)}")

        # Try importing from relative path in case OmniParser is in the same workspace
        try:
            current_dir = Path(__file__).resolve().parent
            workspace_dir = current_dir.parent.parent.parent.parent

            omniparser_dir = workspace_dir / "OmniParser"
            if omniparser_dir.exists():
                spec = importlib.util.spec_from_file_location(
                    "omniparser", str(omniparser_dir / "util" / "omniparser.py")
                )
                if spec and spec.loader:
                    omniparser_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(omniparser_module)
                    return omniparser_module.OmniParser
        except Exception as e:
            logger.error(f"Failed to import OmniParser from relative path: {str(e)}")

        # If all attempts failed, raise an import error
        raise ImportError("Failed to import OmniParser. Please install OmniParser or set OMNIPARSER_SOURCE_PATH.")


def get_omniparser():
    """
    Get or initialize the OmniParser instance.

    Returns:
        OmniParser instance

    Raises:
        ImportError: If OmniParser dependencies are not installed
        RuntimeError: If OmniParser fails to initialize
    """
    global _omniparser_instance
    if _omniparser_instance is None:
        try:
            # Get OmniParser model configuration
            model_manager = ModelManager()
            config = model_manager.get_config()

            # Import OmniParser class
            OmniParser = _import_omniparser()
            if not OmniParser:
                raise ImportError("Failed to import OmniParser class")

            # Initialize OmniParser with configuration
            _omniparser_instance = OmniParser(
                {
                    "som_model_path": config.som_model_path,
                    "caption_model_name": config.caption_model_name,
                    "caption_model_path": config.caption_model_path,
                    "device": config.device,
                    "BOX_THRESHOLD": config.box_threshold,
                }
            )

            logger.info("OmniParser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OmniParser: {str(e)}")
            raise RuntimeError(f"Failed to initialize OmniParser: {str(e)}")

    return _omniparser_instance


@CALLABLES.register("screen.parse_omniparser")
def parse_screen(screen_image: Union[str, bytes, BytesIO]) -> Dict[str, Any]:
    """
    Parse a screen image using OmniParser.

    Args:
        screen_image: The screen image to parse.
            Can be a base64 encoded string, bytes, or BytesIO object.

    Returns:
        Dict containing parsed UI elements and visualization image

    Raises:
        RuntimeError: If OmniParser fails to parse the image
    """
    try:
        omniparser = get_omniparser()

        # Process image format
        if isinstance(screen_image, BytesIO):
            screen_image.seek(0)
            screen_image = screen_image.read()

        if isinstance(screen_image, bytes):
            screen_image = base64.b64encode(screen_image).decode("utf-8")

        # Run OmniParser
        logger.info("Parsing screen with OmniParser")
        dino_labeled_img, parsed_content_list = omniparser.parse(screen_image)

        return {"som_image_base64": dino_labeled_img, "parsed_content_list": parsed_content_list}

    except Exception as e:
        logger.error(f"Error parsing screen with OmniParser: {str(e)}")
        raise RuntimeError(f"Error parsing screen with OmniParser: {str(e)}")


@CALLABLES.register("screen.get_element_by_description")
def get_element_by_description(
    parsed_elements: Dict[str, Any], description: str, similarity_threshold: float = 0.7
) -> Optional[Dict[str, Any]]:
    """
    Find a UI element by its description in parsed elements.

    Args:
        parsed_elements: Result from screen.parse_omniparser
        description: Description of the element to find
        similarity_threshold: Minimum similarity score (0.0-1.0)

    Returns:
        Dictionary containing the element if found, otherwise None
    """
    try:
        # Get text similarity function
        if "text.similarity" in CALLABLES:
            similarity_func = CALLABLES["text.similarity"]
        else:
            # Simple fallback similarity
            def similarity_func(text1, text2):
                normalized1 = text1.lower()
                normalized2 = text2.lower()

                # Exact match
                if normalized1 == normalized2:
                    return 1.0

                # Contains match
                if normalized1 in normalized2 or normalized2 in normalized1:
                    return 0.8

                # Word overlap
                words1 = set(normalized1.split())
                words2 = set(normalized2.split())
                overlap = len(words1.intersection(words2))

                if overlap > 0:
                    return 0.5 + (0.5 * overlap / max(len(words1), len(words2)))

                return 0.0

        # Find best match
        best_match = None
        best_score = 0.0

        for element in parsed_elements.get("parsed_content_list", []):
            if "description" not in element:
                continue

            score = similarity_func(element["description"], description)

            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = element

        if best_match:
            best_match["similarity_score"] = best_score

        return best_match

    except Exception as e:
        logger.error(f"Error finding element by description: {str(e)}")
        return None
