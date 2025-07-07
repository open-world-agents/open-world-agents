from owa.data import load_dataset


def test_list_available():
    datasets = load_dataset.list_available()
    assert isinstance(datasets, list)
    assert any("open-world-agents/example_dataset" in ds for ds in datasets)


def test_load_example_dataset():
    # This will download a small dataset from HuggingFace (network required)
    ds = load_dataset("open-world-agents/example_dataset")
    assert "train" in ds or len(ds) > 0
