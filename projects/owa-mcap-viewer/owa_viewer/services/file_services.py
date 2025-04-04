import os
from pathlib import Path, PurePosixPath

import fsspec
import fsspec.implementations
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem

from ..schema import File, FilePair

from dotenv import load_dotenv

load_dotenv()

EXPORT_PATH = os.environ.get("EXPORT_PATH", "./data")
print(f"{EXPORT_PATH=}")


def list_file(repo_id: str) -> list[File]:
    """For a given repository, list all available data files."""

    if repo_id == "local":
        protocol = "file"
        fs: LocalFileSystem = fsspec.filesystem(protocol=protocol)
        path = EXPORT_PATH
    else:
        protocol = "hf"
        fs: HfFileSystem = fsspec.filesystem(protocol=protocol)
        path = f"datasets/{repo_id}"

    files = []
    for mcap_file in fs.glob(f"{path}/**/*.mcap"):
        mcap_file = PurePosixPath(mcap_file)
        if fs.exists(mcap_file.with_suffix(".mkv")) and fs.exists(mcap_file.with_suffix(".mcap")):
            basename = (mcap_file.parent / mcap_file.stem).as_posix()
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
