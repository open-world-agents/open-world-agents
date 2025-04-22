from typing import Any, Dict, Optional

from fastapi import HTTPException


class AppError(HTTPException):
    """Base application error class"""

    def __init__(
        self, status_code: int, detail: str, error_code: str = "GENERAL_ERROR", extra: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.extra = extra or {}
        super().__init__(status_code=status_code, detail=detail)


class FileNotFoundError(AppError):
    """Raised when a file is not found"""

    def __init__(self, filename: str):
        super().__init__(
            status_code=404,
            detail=f"File not found: {filename}",
            error_code="FILE_NOT_FOUND",
            extra={"filename": filename},
        )


class FileDownloadError(AppError):
    """Raised when a file download fails"""

    def __init__(self, url: str, reason: str):
        super().__init__(
            status_code=500,
            detail=f"Failed to download file: {reason}",
            error_code="FILE_DOWNLOAD_ERROR",
            extra={"url": url, "reason": reason},
        )


class InvalidDataError(AppError):
    """Raised when data is invalid"""

    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail, error_code="INVALID_DATA")


class McapProcessingError(AppError):
    """Raised when MCAP processing fails"""

    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail, error_code="MCAP_PROCESSING_ERROR")
