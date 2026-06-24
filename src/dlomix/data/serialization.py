"""
Persistence for processed peptide datasets: serialize runtime state, save to disk,
and reconstruct the correct subclass on load.

Operates on a ``PeptideDataset`` instance via its public attributes/constants, so it
carries no hard import of the dataset class (``load_processed_dataset`` imports it lazily).
"""

import importlib
import json
from pathlib import Path

from .dataset_config import DatasetConfig

# Transient / non-reconstructable attributes excluded from the saved state.
_STATE_EXCLUDE = {
    "hf_dataset",
    "_config",
    "data_source",
    "_split_mode",
    "_temp_stratify_column",
}


def build_serializable_state(dataset) -> dict:
    """Collect the dataset's runtime attributes as a JSON-serializable dict."""
    state = {}
    for key, value in dataset.__dict__.items():
        if key in _STATE_EXCLUDE:
            continue
        try:
            json.dumps(value)
            state[key] = value
        except (TypeError, ValueError):
            # keep containers as-is (their items round-trip); stringify other objects
            if isinstance(value, dict):
                state[key] = value
            elif isinstance(value, (list, tuple, set)):
                state[key] = list(value)
            else:
                state[key] = str(type(value))
    return state


def save_dataset(dataset, path: str, overwrite: bool = False) -> bool:
    """Save config, runtime-state metadata, and the HF dataset to ``path``."""
    path_obj = Path(path)
    if path_obj.exists() and not overwrite:
        raise FileExistsError(
            f"Directory {path} already exists. Set overwrite=True to overwrite and "
            "replace the saved dataset."
        )
    path_obj.mkdir(parents=True, exist_ok=True)

    # original config (input parameters as provided by the user)
    dataset._config.save_config_json(str(path_obj / dataset.CONFIG_JSON_NAME))

    metadata = {
        "class_name": type(dataset).__name__,
        "module_name": type(dataset).__module__,
        "state": build_serializable_state(dataset),
        "version": dataset.SERIALIZATION_VERSION,
    }
    (path_obj / dataset.METADATA_JSON_NAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if dataset.hf_dataset is not None:
        dataset.hf_dataset.save_to_disk(str(path_obj / "hf_dataset"))

    return True


def validate_loaded_state(dataset) -> None:
    """Raise ValueError if a freshly loaded dataset is inconsistent."""
    if getattr(dataset, "hf_dataset", None) is None:
        raise ValueError("HuggingFace dataset not loaded properly.")

    if not dataset.processed:
        raise ValueError("Dataset should be marked as processed after loading.")

    if hasattr(dataset, "_relevant_columns"):
        expected_columns = set(dataset._relevant_columns)
        for split in dataset.hf_dataset.keys():
            dataset_columns = set(dataset.hf_dataset[split].column_names)
            if not expected_columns.issubset(dataset_columns):
                missing = expected_columns - dataset_columns
                raise ValueError(
                    f"Split '{split}' is missing expected columns: {missing}"
                )


def load_processed_dataset(path: str, validate: bool = True):
    """
    Load a processed peptide dataset from a given path.

    Parameters
    ----------
    path : str
        Path to the peptide dataset.
    validate : bool, optional
        Whether to validate the loaded dataset state, by default True.

    Returns
    -------
    dlomix.data.PeptideDataset or one of its child classes
        Peptide dataset.
    """
    from .dataset import PeptideDataset  # local import to avoid a cycle

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Provided directory for loading the dataset: {path} does not exist."
        )

    config = DatasetConfig.load_config_json(
        str(path_obj / PeptideDataset.CONFIG_JSON_NAME)
    )

    metadata_path = path_obj / PeptideDataset.METADATA_JSON_NAME
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )

    # reconstruct the correct subclass recorded at save time
    module = importlib.import_module("dlomix.data")
    class_name = (
        metadata.get("class_name", "PeptideDataset") if metadata else "PeptideDataset"
    )
    cls = getattr(module, class_name)

    # processed=True skips re-processing during construction
    config.processed = True
    instance = cls.from_dataset_config(config)
    instance._config.processed = False
    instance.processed = True

    if metadata:
        for key, value in metadata["state"].items():
            setattr(instance, key, value)

    hf_dataset_path = path_obj / "hf_dataset"
    if hf_dataset_path.exists():
        from datasets import load_from_disk

        instance.hf_dataset = load_from_disk(str(hf_dataset_path))

    if validate:
        validate_loaded_state(instance)

    return instance
