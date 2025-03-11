"""
API client for OmniParser.

This module provides functionality for connecting to an external OmniParser API server.
"""

import os
import logging
import json
import time
import base64
from typing import Dict, Any, Union, Optional
from io import BytesIO

import requests
from owa.registry import CALLABLES

logger = logging.getLogger(__name__)


class OmniParserAPIConfig:
    """Configuration for OmniParser API client."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/parse/",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize OmniParser API configuration.
        
        Args:
            api_url: URL for the OmniParser API endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay


class OmniParserAPIClient:
    """Client for interacting with OmniParser API server."""
    
    def __init__(self, config: Optional[OmniParserAPIConfig] = None):
        """
        Initialize the OmniParser API client.
        
        Args:
            config: Client configuration, or None to use default/environment values
        """
        self.config = config or self._load_config_from_env()
        self.session = requests.Session()
    
    def _load_config_from_env(self) -> OmniParserAPIConfig:
        """
        Load API configuration from environment variables.
        
        Returns:
            OmniParserAPIConfig with values from environment
        """
        api_url = os.getenv("OMNIPARSER_API_URL", "http://localhost:8000/parse/")
        timeout = int(os.getenv("OMNIPARSER_API_TIMEOUT", "30"))
        max_retries = int(os.getenv("OMNIPARSER_API_MAX_RETRIES", "3"))
        retry_delay = int(os.getenv("OMNIPARSER_API_RETRY_DELAY", "1"))
        
        return OmniParserAPIConfig(
            api_url=api_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def parse_screen(self, screen_image: Union[str, bytes, BytesIO]) -> Dict[str, Any]:
        """
        Parse a screen image using OmniParser API.
        
        Args:
            screen_image: The screen image to parse.
                Can be a base64 encoded string, bytes, or BytesIO object.
                
        Returns:
            Dict containing parsed UI elements and visualization image
            
        Raises:
            ConnectionError: If API server cannot be reached
            TimeoutError: If request times out
            ValueError: If API response format is invalid
            RuntimeError: If API request fails
        """
        # Process image format
        if isinstance(screen_image, BytesIO):
            screen_image.seek(0)
            screen_image = screen_image.read()
            
        if isinstance(screen_image, bytes):
            screen_image = base64.b64encode(screen_image).decode('utf-8')
        
        # Prepare request
        payload = {"base64_image": screen_image}
        
        # Try multiple times with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Calling OmniParser API at {self.config.api_url} (attempt {attempt+1}/{self.config.max_retries})")
                
                response = self.session.post(
                    self.config.api_url,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Validate response format
                        if not isinstance(result, dict) or "som_image_base64" not in result or "parsed_content_list" not in result:
                            raise ValueError("API response format is invalid")
                        return result
                    except json.JSONDecodeError:
                        last_error = ValueError(f"Cannot parse API response as JSON: {response.text[:100]}...")
                        raise last_error
                else:
                    error_msg = f"API request failed: HTTP {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail.get('error', '')}"
                    except:
                        error_msg += f" - {response.text[:100]}..."
                    
                    # Only retry for server errors (5xx)
                    if 500 <= response.status_code < 600 and attempt < self.config.max_retries - 1:
                        logger.warning(f"{error_msg}, retrying...")
                    else:
                        last_error = RuntimeError(error_msg)
                        raise last_error
                        
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Connection error, retrying in {self.config.retry_delay}s...")
                else:
                    raise ConnectionError(f"Cannot connect to OmniParser API server at {self.config.api_url}")
                    
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Request timeout, retrying in {self.config.retry_delay}s...")
                else:
                    raise TimeoutError(f"OmniParser API request timed out after {self.config.timeout}s")
            
            except (ValueError, RuntimeError):
                # Don't retry for client errors
                raise
                
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Error: {str(e)}, retrying in {self.config.retry_delay}s...")
                else:
                    raise RuntimeError(f"OmniParser API request failed: {str(e)}")
            
            # Delay before retry with exponential backoff
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (2 ** attempt))
        
        # If we reach here, all retries failed
        if last_error:
            raise last_error
        else:
            # This should never happen because we would have raised an exception earlier
            raise RuntimeError("OmniParser API request failed after all retries for unknown reason")


@CALLABLES.register("screen.parse_omniparser_api")
def parse_screen_api(screen_image: Union[str, bytes, BytesIO]) -> Dict[str, Any]:
    """
    Parse a screen image using OmniParser API server.
    
    Args:
        screen_image: The screen image to parse.
            Can be a base64 encoded string, bytes, or BytesIO object.
            
    Returns:
        Dict containing parsed UI elements and visualization image
        
    Raises:
        Various exceptions if API request fails
    """
    client = OmniParserAPIClient()
    return client.parse_screen(screen_image) 