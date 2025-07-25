import pytest

from owa.data import load_dataset


@pytest.mark.network
def test_list_available():
    datasets = load_dataset.list_available()
    assert isinstance(datasets, list)
    assert any("open-world-agents/example_dataset" in ds for ds in datasets)


@pytest.mark.network
def test_load_example_dataset():
    # This will download a small dataset from HuggingFace (network required)
    # For now, catch NotImplementedError
    with pytest.raises(NotImplementedError):
        ds = load_dataset("open-world-agents/example_dataset")  # noqa: F841
    # assert "train" in ds or len(ds) > 0
