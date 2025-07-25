"""Dataset discovery and remote loading utilities."""

import json
import posixpath
from typing import List, Optional

import fsspec
from datasets.utils.file_utils import url_to_fs
from datasets.utils.typing import PathLike
from huggingface_hub import list_datasets as hf_list_datasets


def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """Check if filesystem is remote (not local)."""
    try:
        from fsspec.implementations.local import LocalFileSystem

        return not isinstance(fs, LocalFileSystem)
    except ImportError:
        # Fallback: check if it's a local filesystem by protocol
        return getattr(fs, "protocol", None) not in ("file", None)


def list_datasets(format_filter: str = "OWA") -> List[str]:
    """
    List available OWA datasets on HuggingFace Hub.

    This function searches for datasets tagged with the specified format
    and returns their repository IDs for easy loading.

    Args:
        format_filter: Filter datasets by format tag (default: "OWA")

    Returns:
        List of dataset repository IDs

    Example:
        ```python
        from owa.data.datasets import list_datasets

        # List all OWA datasets
        datasets = list_datasets()
        print(f"Available OWA datasets: {datasets}")

        # Load a specific dataset
        dataset = load_dataset(datasets[0])
        ```
    """
    try:
        # List datasets on HuggingFace with the format filter
        results = hf_list_datasets(filter=format_filter)
        # Return repo_ids only
        return [ds.id for ds in results]
    except Exception as e:
        print(f"Warning: Could not list datasets from HuggingFace Hub: {e}")
        return []


def load_config_from_path(config_path: str, fs: fsspec.AbstractFileSystem) -> Optional[dict]:
    """
    Load OWA config from a path, supporting both local and remote filesystems.

    Args:
        config_path: Path to the owa_config.json file
        fs: Filesystem instance (local or remote)

    Returns:
        Config dictionary if found, None otherwise
    """
    try:
        if fs.isfile(config_path):
            if is_remote_filesystem(fs):
                # For remote filesystems, read the file content directly
                with fs.open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            else:
                # For local filesystems, use standard file operations
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            return config_data
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")

    return None


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
