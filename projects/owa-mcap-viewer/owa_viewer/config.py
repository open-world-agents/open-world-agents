import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Centralized application settings"""

    # File storage settings
    EXPORT_PATH: str = os.environ.get("EXPORT_PATH", "./data")
    PUBLIC_HOSTING_MODE: bool = not bool(os.environ.get("EXPORT_PATH"))

    # Cache settings
    # MCAP_CACHE_SIZE_LIMIT: int = int(os.environ.get("MCAP_CACHE_SIZE_LIMIT", 1024 * 1024 * 1024))  # 1GB default, not implemented yet
    FILE_CACHE_TTL: int = int(os.environ.get("FILE_CACHE_TTL", 600))  # 10 min default
    DEFAULT_CACHE_TTL: int | None = int(os.environ.get("DEFAULT_CACHE_TTL", 3600))  # 1 hour default
    CACHE_DIR: str = os.environ.get("CACHE_DIR", "./cache")

    # Server settings
    PORT: int = int(os.environ.get("PORT", 7860))
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    # File patterns
    FEATURED_DATASETS: list[str] = [
        "local",
        "open-world-agents/example_dataset",
        "open-world-agents/example_dataset2",
        "open-world-agents/example-djmax",
        "open-world-agents/example-aimlab",
        "open-world-agents/example-pubg-battleground",
    ]

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure cache directory exists
Path(settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)
