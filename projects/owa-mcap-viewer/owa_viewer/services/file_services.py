import os
from pathlib import Path

import fsspec
import fsspec.implementations
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from ..schema import File, FilePair

EXPORT_PATH = os.environ.get("EXPORT_PATH", "./data")


def list_file(repo_id: str) -> list[File]:
    """For a given repository, list all available data files."""
    repo_id = "local"  # TODO: remove this line

    if repo_id == "local":
        protocol = "file"
        fs: LocalFileSystem = fsspec.filesystem(protocol=protocol)
        path = EXPORT_PATH
    else:
        protocol = "hf"
        fs: HfFileSystem = fsspec.filesystem(protocol=protocol)
        path = f"datasets/{repo_id}"
    # fs, path = url_to_fs("hf://datasets/open-world-agents/example_dataset/")

    files = []
    for mcap_file in fs.glob(f"{path}/**/*.mcap"):
        mcap_file = Path(mcap_file)
        basename = (mcap_file.parent / mcap_file.stem).as_posix()
        if fs.exists(mcap_file.with_suffix(".mkv")) and fs.exists(mcap_file.with_suffix(".mcap")):
            if protocol == "file":
                basename = Path(basename).relative_to(EXPORT_PATH).as_posix()
                item = f"/files/{basename}"
            else:
                basename = basename[len(f"datasets/{repo_id}/") :]
                item = f"https://huggingface.co/{repo_id}/resolve/main/{basename}"
            files.append(File(name=basename, url=item, size=fs.info(mcap_file).get("size", 0)))
    return files


def list_filepair(repo_id: str) -> list[FilePair]:
    file_pairs = []
    files = list_file(repo_id)
    for file in files:
        file_pairs.append(FilePair(mcap_file=file.name + ".mcap", mkv_file=file.name + ".mkv", basename=file.name))
    return file_pairs


if __name__ == "__main__":
    print(list_file("local"))
    print(list_file("open-world-agents/example_dataset"))
