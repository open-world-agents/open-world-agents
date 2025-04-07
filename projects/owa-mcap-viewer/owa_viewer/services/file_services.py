import logging
import os
from pathlib import Path, PurePosixPath

import fsspec
import fsspec.implementations
from dotenv import load_dotenv
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem
from rich import print

from ..schema import OWAFile

load_dotenv()

EXPORT_PATH = os.environ.get("EXPORT_PATH", "./data")
EXPORT_PATH = Path(EXPORT_PATH).as_posix()

logger = logging.getLogger(__name__)
logger.info(f"{EXPORT_PATH=}")


def safe_join(base_dir: str, *paths: str) -> Path | None:
    """Join paths and ensure the result is within the base directory."""
    base = Path(base_dir).resolve()
    target = (base / Path(*paths)).resolve()

    if not str(target).startswith(str(base)):
        logger.error(f"Unsafe path: {target} is outside of base directory {base}")
        return None

    return target


def list_file(repo_id: str) -> list[OWAFile]:
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
            if repo_id == "local":
                basename = PurePosixPath(basename).relative_to(EXPORT_PATH).as_posix()
                url = f"{basename}"
                local = True
            else:
                basename = basename[len(f"datasets/{repo_id}/") :]
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{basename}"
                local = False
            files.append(
                OWAFile(
                    basename=basename,
                    url=url,
                    size=fs.info(mcap_file).get("size", 0),
                    local=local,
                    url_mcap=url + ".mcap",
                    url_mkv=url + ".mkv",
                )
            )
    return files


if __name__ == "__main__":
    print(list_file("local"))
    print(list_file("open-world-agents/example_dataset"))
