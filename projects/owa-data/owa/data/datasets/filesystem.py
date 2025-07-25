"""Filesystem utilities for OWA datasets."""

import json
import posixpath
from typing import Optional

import fsspec
from datasets.utils.file_utils import url_to_fs
from datasets.utils.typing import PathLike


def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """Check if filesystem is remote (not local)."""
    try:
        from fsspec.implementations.local import LocalFileSystem

        return not isinstance(fs, LocalFileSystem)
    except ImportError:
        # Fallback: check if it's a local filesystem by protocol
        return getattr(fs, "protocol", None) not in ("file", None)


def load_config_from_path(config_path: str, fs: fsspec.AbstractFileSystem) -> dict:
    """
    Load OWA config from a path, supporting both local and remote filesystems.

    Args:
        config_path: Path to the owa_config.json file
        fs: Filesystem instance (local or remote)
    """

    if not fs.isfile(config_path):
        raise FileNotFoundError(
            f"Config file does not exist or is not a file: {config_path} (protocol: {getattr(fs, 'protocol', 'unknown')})"
        )

    if is_remote_filesystem(fs):
        # For remote filesystems, read the file content directly
        with fs.open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    else:
        # For local filesystems, use standard file operations
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    return config_data


def resolve_dataset_path_and_config(dataset_path: PathLike, storage_options: Optional[dict] = None) -> tuple:
    """
    Resolve dataset path and load config, supporting both local and remote filesystems.

    Args:
        dataset_path: Path to dataset directory (local or remote)
        storage_options: Options for remote filesystem access

    Returns:
        Tuple of (resolved_path, config_dict, filesystem)
    """
    # Get filesystem and resolve path
    fs, resolved_path = url_to_fs(dataset_path, **(storage_options or {}))

    # Try to load OWA config
    config_path = posixpath.join(resolved_path, "owa_config.json")
    config_data = load_config_from_path(config_path, fs)

    return resolved_path, config_data, fs
