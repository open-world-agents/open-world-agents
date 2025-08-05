"""Utility functions for transforms."""

import os
from typing import Optional


def resolve_episode_path(episode_path: str, mcap_root_directory: Optional[str] = None) -> str:
    """Resolve episode path, raising error if relative path needs mcap_root_directory."""
    if not episode_path or os.path.isabs(episode_path):
        return episode_path

    if not mcap_root_directory:
        raise ValueError(f"mcap_root_directory required for relative path: '{episode_path}'")

    return os.path.join(mcap_root_directory, episode_path)
