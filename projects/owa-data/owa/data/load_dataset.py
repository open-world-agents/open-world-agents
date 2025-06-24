import datasets
from huggingface_hub import list_datasets


class LoadDataset:
    @staticmethod
    def list_available(format: str = "OWA"):
        # List datasets on HuggingFace with the OWA tag
        # Optionally filter by format (not strictly enforced, but can be used for future extension)
        results = list_datasets(filter=format)
        # Return repo_ids only
        return [ds.id for ds in results]

    def __call__(self, repo_id: str, **kwargs):
        # Load the dataset from HuggingFace using datasets.load_dataset
        # kwargs can be used to pass split, streaming, etc.
        return datasets.load_dataset(repo_id, **kwargs)


# Singleton instance for import style: from owa.data import load_dataset
load_dataset = LoadDataset()
