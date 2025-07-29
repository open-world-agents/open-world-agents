import pytest

from owa.data.datasets import list_datasets, load_from_disk


@pytest.mark.network
def test_list_datasets():
    datasets = list_datasets()
    assert isinstance(datasets, list)
    assert any("open-world-agents/example_dataset" in ds for ds in datasets)


@pytest.mark.network
def test_load_example_dataset():
    # This will download a small dataset from HuggingFace (network required)
    # Test actual loading functionality
    ds = load_from_disk("open-world-agents/example_dataset")
    assert ds is not None
    assert len(ds) > 0
