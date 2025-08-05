"""Transform module for OWA datasets."""

from .event import create_event_transform
from .binned import create_binned_transform
from .tokenized import create_tokenized_transform
from .fsl import create_fsl_transform, FSLTransform, FSLTransformConfig
from .utils import resolve_episode_path


def create_transform(stage: str, mcap_root_directory: str, **kwargs):
    """Create a transform function for a given stage."""
    from ..config import DatasetStage

    if stage == DatasetStage.EVENT:
        return create_event_transform(mcap_root_directory=mcap_root_directory, **kwargs)
    elif stage == DatasetStage.BINNED:
        return create_binned_transform(mcap_root_directory=mcap_root_directory, **kwargs)
    elif stage == DatasetStage.TOKENIZED:
        return create_tokenized_transform(**kwargs)
    elif stage == DatasetStage.FSL:
        return create_fsl_transform(mcap_root_directory=mcap_root_directory, **kwargs)
    else:
        raise ValueError(f"Unknown dataset stage: {stage}")


__all__ = [
    "create_event_transform",
    "create_binned_transform",
    "create_tokenized_transform",
    "create_fsl_transform",
    "create_transform",
    "FSLTransform",
    "FSLTransformConfig",
    "resolve_episode_path",
]
