from loguru import logger

from .encoders import create_encoder
from .processing import create_resampler

# Disable logger by default for library usage (following loguru best practices)
# Reference: https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.data")

__all__ = ["create_encoder", "create_resampler"]
