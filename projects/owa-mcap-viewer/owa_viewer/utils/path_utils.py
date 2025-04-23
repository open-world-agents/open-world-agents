import re
from pathlib import Path
from typing import Optional


def safe_join(base_dir: str, *paths: str) -> Optional[Path]:
    """
    Join paths safely, ensuring the result is within the base directory.

    Args:
        base_dir: The base directory
        paths: Path components to join

    Returns:
        Path object if safe, None if path would escape base directory
    """
    base = Path(base_dir).resolve()

    # Reject paths with suspicious components
    for part in paths:
        if not part or ".." in part.split("/") or part.startswith("/"):
            return None

    # Join and validate
    try:
        target = (base / Path(*paths)).resolve()

        # Ensure the target is a subpath of base_dir
        if not str(target).startswith(str(base)):
            return None

        return target
    except (ValueError, TypeError):
        return None


def extract_original_filename(filename: str) -> Optional[str]:
    """
    Extract original filename from a UUID-appended filename.

    Args:
        filename: Filename that may contain UUID suffix

    Returns:
        Original filename if UUID pattern found, None otherwise
    """
    uuid_pattern = r"(.+)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    match = re.search(uuid_pattern, filename)
    if match:
        return match.group(1)
    return None
