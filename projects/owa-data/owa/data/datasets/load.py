import posixpath
from typing import Optional, Union

import fsspec
from datasets import config
from datasets.utils.typing import PathLike
from fsspec.core import url_to_fs

from .dataset import Dataset, DatasetDict


def load_dataset(path: str):
    raise NotImplementedError("Not implemented yet")


# Copied from: https://github.com/huggingface/datasets/blob/main/src/datasets/load.py#L1429-L1476
def load_from_disk(
    dataset_path: PathLike, keep_in_memory: Optional[bool] = None, storage_options: Optional[dict] = None
) -> Union[Dataset, DatasetDict]:
    """
    Loads a dataset that was previously saved using [`~Dataset.save_to_disk`] from a dataset directory, or
    from a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

    Args:
        dataset_path (`path-like`):
            Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3://my-bucket/dataset/train"`)
            of the [`Dataset`] or [`DatasetDict`] directory where the dataset/dataset-dict will be
            loaded from.
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.

        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the file-system backend, if any.

            <Added version="2.9.0"/>

    Returns:
        [`Dataset`] or [`DatasetDict`]:
        - If `dataset_path` is a path of a dataset directory: the dataset requested.
        - If `dataset_path` is a path of a dataset dict directory, a [`DatasetDict`] with each split.

    Example:

    ```py
    >>> from datasets import load_from_disk
    >>> ds = load_from_disk('path/to/dataset/directory')
    ```
    """
    fs: fsspec.AbstractFileSystem
    fs, *_ = url_to_fs(dataset_path, **(storage_options or {}))
    if not fs.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    if fs.isfile(posixpath.join(dataset_path, config.DATASET_INFO_FILENAME)) and fs.isfile(
        posixpath.join(dataset_path, config.DATASET_STATE_JSON_FILENAME)
    ):
        return Dataset.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    elif fs.isfile(posixpath.join(dataset_path, config.DATASETDICT_JSON_FILENAME)):
        return DatasetDict.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    else:
        raise FileNotFoundError(
            f"Directory {dataset_path} is neither a `Dataset` directory nor a `DatasetDict` directory."
        )
