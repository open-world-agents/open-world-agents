"""Dataset discovery utilities."""

from typing import List

from huggingface_hub import list_datasets as hf_list_datasets


def list_datasets(format_filter: str = "OWA") -> List[str]:
    """
    List available OWA datasets on HuggingFace Hub.

    This function searches for datasets tagged with the specified format
    and returns their repository IDs for easy loading.

    Args:
        format_filter: Filter datasets by format tag (default: "OWA")

    Returns:
        List of dataset repository IDs

    Example:
        ```python
        from owa.data.datasets import list_datasets

        # List all OWA datasets
        datasets = list_datasets()
        print(f"Available OWA datasets: {datasets}")

        # Load a specific dataset
        dataset = load_from_disk(datasets[0])
        ```
    """
    try:
        # List datasets on HuggingFace with the format filter
        results = hf_list_datasets(filter=format_filter)
        # Return repo_ids only
        return [ds.id for ds in results]
    except Exception as e:
        print(f"Warning: Could not list datasets from HuggingFace Hub: {e}")
        return []
