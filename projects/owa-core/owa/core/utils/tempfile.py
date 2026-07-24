"""
Enhanced temporary file utilities with cross-platform compatibility.

This module provides a cross-platform NamedTemporaryFile that works reliably
on Windows by handling file locking issues.
"""

import tempfile as _tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


@contextmanager
def NamedTemporaryFile(
    mode: str = "w+b",
    buffering: int = -1,
    encoding: Optional[str] = None,
    newline: Optional[str] = None,
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[str] = None,
    delete: bool = True,
    *,
    errors: Optional[str] = None,
):
    """
    Create a named temporary file with cross-platform compatibility.

    This is a drop-in replacement for tempfile.NamedTemporaryFile that works
    reliably on Windows by handling file locking issues. The API is consistent
    with the standard library version.

    Args:
        mode: File mode (default: "w+b")
        buffering: Buffer size (default: -1)
        encoding: Text encoding (default: None)
        newline: Newline handling (default: None)
        suffix: File suffix (default: None)
        prefix: File prefix (default: None)
        dir: Directory to create file in (default: None)
        delete: Whether to delete file on close (default: True)
        errors: Error handling mode (default: None)

    Yields:
        A file-like object with a .name attribute containing the file path

    Note:
        On Windows, this handles the file locking issue by creating the file
        with delete=False and manually cleaning up afterward.
        See: https://stackoverflow.com/a/23212515
    """
    # Create temporary file with delete=False + .close() to avoid Windows locking issues
    temp_file = _tempfile.NamedTemporaryFile(
        mode=mode,
        buffering=buffering,
        encoding=encoding,
        newline=newline,
        suffix=suffix,
        prefix=prefix,
        dir=dir,
        delete=False,
        errors=errors,
    )
    temp_file.close()

    try:
        yield temp_file
    finally:
        # Clean up the file if delete was requested
        if delete:
            Path(temp_file.name).unlink(missing_ok=True)


# Direct re-exports from standard tempfile module
mkstemp = _tempfile.mkstemp
mkdtemp = _tempfile.mkdtemp
gettempdir = _tempfile.gettempdir
gettempprefix = _tempfile.gettempprefix
TemporaryFile = _tempfile.TemporaryFile
TemporaryDirectory = _tempfile.TemporaryDirectory
SpooledTemporaryFile = _tempfile.SpooledTemporaryFile
template = _tempfile.template
