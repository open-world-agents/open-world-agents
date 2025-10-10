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
        list: List of successful serialized objects. Each element is a tuple of
              (serializer_name, serialized_data) for successful serializations.
    """
    successful_pickles = []

    # Try pickle first
    try:
        serialized_data = pickle.dumps(obj)
        successful_pickles.append(("pickle", serialized_data))
    except Exception:
        pass

    # Try dill if available
    if HAS_DILL and dill is not None:
        try:
            serialized_data = dill.dumps(obj)
            successful_pickles.append(("dill", serialized_data))
        except Exception:
            pass

    return successful_pickles


class TestTransformSerialization:
    """Test serialization of all dataset transforms."""

    def test_event_transform_serialization(self, record_property):
        """Test event transform serialization with pickle/dill."""
        transform = create_event_transform(
            encoder_type="factorized", load_images=True, mcap_root_directory="/test/mcap"
        )

        # Must be serializable with either pickle or dill
        successful_pickles = can_serialize_with_either(transform)
        method_names = [method for method, _ in successful_pickles]

        # Record which methods succeeded for test reporting
        record_property("serialization_methods", ", ".join(method_names) if method_names else "none")

        assert len(successful_pickles) > 0, (
            f"Transform should be serializable with at least one method. Tried: pickle, dill. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
        )

    def test_binned_transform_serialization(self, record_property):
        """Test binned transform serialization with pickle/dill."""
        transform = create_binned_transform(
            instruction="Complete the computer task",
            encoder_type="factorized",
            load_images=True,
            encode_actions=True,
            mcap_root_directory="/test/mcap",
        )

        # Must be serializable with either pickle or dill
        successful_pickles = can_serialize_with_either(transform)
        method_names = [method for method, _ in successful_pickles]

        # Record which methods succeeded for test reporting
        record_property("serialization_methods", ", ".join(method_names) if method_names else "none")

        assert len(successful_pickles) > 0, (
            f"Transform should be serializable with at least one method. Tried: pickle, dill. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
        )

    def test_tokenized_transform_serialization(self, record_property):
        """Test tokenized transform serialization with pickle/dill."""
        transform = create_tokenized_transform()

        # Must be serializable with either pickle or dill
        successful_pickles = can_serialize_with_either(transform)
        method_names = [method for method, _ in successful_pickles]

        # Record which methods succeeded for test reporting
        record_property("serialization_methods", ", ".join(method_names) if method_names else "none")

        assert len(successful_pickles) > 0, (
            f"Transform should be serializable with at least one method. Tried: pickle, dill. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
        )

    def test_fsl_transform_serialization(self, record_property):
        """Test FSL transform serialization (function, config, and class)."""
        # Test 1: Function creation
        transform_func = create_fsl_transform(load_images=True, mcap_root_directory="/test/mcap", pad_token_id=42)
        successful_pickles = can_serialize_with_either(transform_func)
        method_names = [method for method, _ in successful_pickles]

        record_property("fsl_function_serialization_methods", ", ".join(method_names) if method_names else "none")
        assert len(successful_pickles) > 0, (
            f"FSL function should be serializable. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
        )

        # Verify function is callable after deserialization
        serialized = pickle.dumps(transform_func)
        deserialized = pickle.loads(serialized)
        assert callable(deserialized)

        # Test 2: Config serialization
        config = FSLTransformConfig(load_images=True, mcap_root_directory="/test/mcap", pad_token_id=42)
        successful_pickles = can_serialize_with_either(config)
        method_names = [method for method, _ in successful_pickles]

        record_property("fsl_config_serialization_methods", ", ".join(method_names) if method_names else "none")
        assert len(successful_pickles) > 0, (
            f"FSL config should be serializable. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
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
        successful_pickles = can_serialize_with_either(transform_none_config)
        method_names = [method for method, _ in successful_pickles]

        record_property("fsl_none_config_serialization_methods", ", ".join(method_names) if method_names else "none")
        assert len(successful_pickles) > 0, (
            f"FSL transform with None config should be serializable. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
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
            successful_pickles = can_serialize_with_either(transform)
            method_names = [method for method, _ in successful_pickles]
            all_results[stage.name] = method_names

            assert len(successful_pickles) > 0, (
                f"{stage} transform should be serializable with at least one method. Tried: pickle, dill. Succeeded with: {', '.join(method_names) if method_names else 'none'}"
            )

        # Record results for all stages
        for stage_name, methods in all_results.items():
            record_property(f"{stage_name.lower()}_serialization_methods", ", ".join(methods) if methods else "none")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
