"""
OmniParser model management utilities.

This module provides functionality for managing OmniParser models,
including downloading, versioning, and configuration.
"""

import os
import json
import logging
import hashlib
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class OmniParserModelConfig:
    """Configuration for OmniParser models."""
    
    def __init__(
        self,
        som_model_path: Optional[str] = None,
        caption_model_name: str = "florence2",
        caption_model_path: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.05
    ):
        """
        Initialize OmniParser model configuration.
        
        Args:
            som_model_path: Path to the SOM detection model weights
            caption_model_name: Name of the caption model to use
            caption_model_path: Path to the caption model weights
            device: Device to use for inference ("cuda" or "cpu")
            box_threshold: Threshold for detection boxes
        """
        self.som_model_path = som_model_path
        self.caption_model_name = caption_model_name
        self.caption_model_path = caption_model_path
        self.device = device
        self.box_threshold = box_threshold


class ModelManager:
    """Manager for OmniParser models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or os.getenv(
            "OMNIPARSER_CONFIG_PATH",
            "~/.owa/models/omniparser.json"
        )
        self.config_path = Path(self.config_path).expanduser()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_dir = Path(os.getenv(
            "OMNIPARSER_MODEL_DIR",
            "~/.owa/models/omniparser"
        )).expanduser()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration or create default.
        
        Returns:
            Dict containing model configuration
        """
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)

        # Default configuration
        default_config = {
            "models": {
                "som_detection": {
                    "version": "1.0.0",
                    "url": "https://github.com/microsoft/OmniParser/releases/download/v1.0.0/icon_detect_model.pt",
                    "hash": "sha256:PLACEHOLDER_HASH",  # Replace with actual hash when available
                    "local_path": None
                },
                "florence_caption": {
                    "version": "2.0",
                    "url": "https://github.com/microsoft/OmniParser/releases/download/v1.0.0/florence_caption_model.zip",
                    "hash": "sha256:PLACEHOLDER_HASH",  # Replace with actual hash when available
                    "local_path": None
                }
            },
            "last_update_check": None
        }

        # Save default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def _save_config(self):
        """Save the current configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def download_model(self, model_name: str) -> str:
        """
        Download and install a model.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Path to the downloaded model
            
        Raises:
            ValueError: If model_name is unknown or hash verification fails
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = self.config["models"][model_name]
        url = model_info["url"]

        # Set local path
        filename = url.split("/")[-1]
        local_path = self.model_dir / filename

        logger.info(f"Downloading {model_name} from {url}")

        # Download file
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(local_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=4096):
                size = f.write(data)
                bar.update(size)

        # Verify hash if available
        if "hash" in model_info and model_info["hash"] != "sha256:PLACEHOLDER_HASH":
            hash_algo, expected_hash = model_info["hash"].split(":", 1)
            if hash_algo == "sha256":
                file_hash = hashlib.sha256()
                with open(local_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        file_hash.update(chunk)

                if file_hash.hexdigest() != expected_hash:
                    local_path.unlink()  # Delete file
                    raise ValueError(f"Hash mismatch for {model_name}")

        # Update configuration
        model_info["local_path"] = str(local_path)
        self._save_config()

        return str(local_path)

    def get_model_path(self, model_name: str, download_if_missing: bool = True) -> Optional[str]:
        """
        Get the path to a model, downloading it if necessary.
        
        Args:
            model_name: Name of the model
            download_if_missing: Whether to download the model if it's missing
            
        Returns:
            Path to the model, or None if not available and not downloaded
            
        Raises:
            ValueError: If model_name is unknown
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = self.config["models"][model_name]

        if model_info["local_path"] and Path(model_info["local_path"]).exists():
            return model_info["local_path"]

        if download_if_missing:
            return self.download_model(model_name)

        return None

    def get_config(self) -> OmniParserModelConfig:
        """
        Get OmniParser model configuration from environment and stored settings.
        
        Returns:
            OmniParserModelConfig object with configured settings
        """
        # Environment variable overrides
        som_model_path = os.getenv("OMNIPARSER_SOM_MODEL_PATH")
        if not som_model_path:
            som_model_path = self.get_model_path("som_detection")
            
        caption_model_path = os.getenv("OMNIPARSER_CAPTION_MODEL_PATH")
        if not caption_model_path:
            caption_model_path = self.get_model_path("florence_caption")
            
        device = os.getenv("OMNIPARSER_DEVICE", "cuda")
        caption_model_name = os.getenv("OMNIPARSER_CAPTION_MODEL", "florence2")
        box_threshold = float(os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.05"))
        
        return OmniParserModelConfig(
            som_model_path=som_model_path,
            caption_model_name=caption_model_name,
            caption_model_path=caption_model_path,
            device=device,
            box_threshold=box_threshold
        ) 