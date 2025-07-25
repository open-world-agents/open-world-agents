from loguru import logger

from .encoders import create_encoder
from .load_dataset import load_dataset
from .transforms import (
    create_binned_dataset_transform,
    create_event_dataset_transform,
)

# Disable logger by default for library usage (following loguru best practices)
# Reference: https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
logger.disable("owa.data")

__all__ = ["load_dataset", "create_encoder", "create_event_dataset_transform", "create_binned_dataset_transform"]
