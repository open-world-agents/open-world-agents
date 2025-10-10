#!/usr/bin/env python3
"""Test serialization of dataset transforms in owa.data."""

import pickle
from unittest.mock import Mock

import pytest

try:
    import dill

    HAS_DILL = True
except ImportError:
    HAS_DILL = False
    dill = None

from owa.data.datasets import DatasetStage, create_transform
from owa.data.datasets.transforms import (
    FSLTransform,
    FSLTransformConfig,
    create_binned_transform,
    create_event_transform,
    create_fsl_transform,
    create_tokenized_transform,
)


def can_serialize_with_either(obj):
    """Test if object can be serialized with either pickle or dill.

    Returns:
        tuple: A tuple containing:
            - list: A list of names of successful serialization methods.
            - dict: A dictionary of exceptions encountered, keyed by serializer name.
    """
    successful_methods = []
    errors = {}

    # Try pickle first
    try:
        pickle.dumps(obj)
        successful_methods.append("pickle")
    except Exception as e:
        errors["pickle"] = e

    # Try dill if available
    if HAS_DILL and dill is not None:
        try:
            dill.dumps(obj)
            successful_methods.append("dill")
        except Exception as e:
            errors["dill"] = e

    return successful_methods, errors


class TestTransformSerialization:
    """Test serialization of all dataset transforms."""

    @pytest.mark.parametrize(
        "transform_name, create_transform_fn",
        [
            (
                "event",
                lambda: create_event_transform(
                    encoder_type="factorized", load_images=True, mcap_root_directory="/test/mcap"
                ),
            ),
            (
                "binned",
                lambda: create_binned_transform(
                    instruction="Complete the computer task",
                    encoder_type="factorized",
                    load_images=True,
                    encode_actions=True,
                    mcap_root_directory="/test/mcap",
                ),
            ),
            ("tokenized", create_tokenized_transform),
        ],
        ids=["event_transform", "binned_transform", "tokenized_transform"],
    )
    def test_transform_serialization(self, record_property, transform_name, create_transform_fn):
        """Test various transform serializations with pickle/dill."""
        transform = create_transform_fn()

        # Must be serializable with either pickle or dill
        successful_methods, errors = can_serialize_with_either(transform)

        # Record which methods succeeded for test reporting
        record_property(
            f"{transform_name}_serialization_methods", ", ".join(successful_methods) if successful_methods else "none"
        )

        assert successful_methods, (
            f"{transform_name.capitalize()} transform should be serializable with at least one method. "
            f"Succeeded with: {', '.join(successful_methods) if successful_methods else 'none'}. "
            f"Errors: {errors}"
        )

    def test_fsl_transform_serialization(self, record_property):
        """Test FSL transform serialization (function, config, and class)."""
        # Test 1: Function creation
        transform_func = create_fsl_transform(load_images=True, mcap_root_directory="/test/mcap", pad_token_id=42)
        successful_methods, errors = can_serialize_with_either(transform_func)

        record_property(
            "fsl_function_serialization_methods", ", ".join(successful_methods) if successful_methods else "none"
        )
        assert successful_methods, (
            f"FSL function should be serializable. Succeeded with: {', '.join(successful_methods) if successful_methods else 'none'}. "
            f"Errors: {errors}"
        )

        # Verify function is callable after deserialization
        serialized = pickle.dumps(transform_func)
        deserialized = pickle.loads(serialized)
        assert callable(deserialized)

        # Test 2: Config serialization
        config = FSLTransformConfig(load_images=True, mcap_root_directory="/test/mcap", pad_token_id=42)
        successful_methods, errors = can_serialize_with_either(config)

        record_property(
            "fsl_config_serialization_methods", ", ".join(successful_methods) if successful_methods else "none"
        )
        assert successful_methods, (
            f"FSL config should be serializable. Succeeded with: {', '.join(successful_methods) if successful_methods else 'none'}. "
            f"Errors: {errors}"
        )

        # Verify config data preservation
        serialized = pickle.dumps(config)
        deserialized = pickle.loads(serialized)
        assert isinstance(deserialized, FSLTransformConfig)
        assert deserialized.load_images == config.load_images
        assert deserialized.mcap_root_directory == config.mcap_root_directory
        assert deserialized.pad_token_id == config.pad_token_id

    def test_fsl_transform_edge_cases(self, record_property):
        """Test FSL transform edge cases and failure scenarios."""
        # Test 1: Mock processor should fail serialization
        mock_processor = Mock()
        mock_processor.is_fast = True
        mock_processor.__class__.__name__ = "MockImageProcessor"

        config = FSLTransformConfig(load_images=True, mcap_root_directory="/test/mcap")
        transform_with_mock = FSLTransform(config=config, image_processor=mock_processor)

        # Should fail with pickle due to Mock object
        with pytest.raises(Exception):
            pickle.dumps(transform_with_mock)

        # Test 2: None config should work (uses default)
        transform_none_config = FSLTransform(config=None)
        successful_methods, errors = can_serialize_with_either(transform_none_config)

        record_property(
            "fsl_none_config_serialization_methods", ", ".join(successful_methods) if successful_methods else "none"
        )
        assert successful_methods, (
            f"FSL transform with None config should be serializable. Succeeded with: {', '.join(successful_methods) if successful_methods else 'none'}. "
            f"Errors: {errors}"
        )

        # Verify None config creates default config
        serialized = pickle.dumps(transform_none_config)
        deserialized = pickle.loads(serialized)
        assert isinstance(deserialized, FSLTransform)
        assert isinstance(deserialized.config, FSLTransformConfig)

    def test_all_stage_transforms_serialization(self, record_property):
        """Test serialization for all stage transforms."""
        stages = [DatasetStage.EVENT, DatasetStage.BINNED, DatasetStage.TOKENIZED, DatasetStage.FSL]
        all_results = {}

        for stage in stages:
            transform = create_transform(stage, "/test/mcap")
            # All transforms must be serializable with either pickle or dill
            successful_methods, errors = can_serialize_with_either(transform)
            all_results[stage.name] = successful_methods

            assert successful_methods, (
                f"{stage} transform should be serializable with at least one method. "
                f"Succeeded with: {', '.join(successful_methods) if successful_methods else 'none'}. "
                f"Errors: {errors}"
            )

        # Record results for all stages
        for stage_name, methods in all_results.items():
            record_property(f"{stage_name.lower()}_serialization_methods", ", ".join(methods) if methods else "none")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
