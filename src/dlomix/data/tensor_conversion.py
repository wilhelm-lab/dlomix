"""
Shared helpers to convert a processed HuggingFace ``Dataset`` into backend tensors.

These are used both by :class:`~dlomix.data.dataset.PeptideDataset` (per split, with
labels) and by :class:`~dlomix.data.inference.PeptidePreprocessor` (inference, no
labels). Kept free of dlomix-internal imports to avoid circular dependencies.
"""

from typing import Iterable, List, Optional, Union

from datasets import Dataset, Sequence, Value


def cast_feature_columns_to_float(
    dataset: Dataset,
    feature_columns: Iterable[str],
    num_proc: Optional[int] = None,
    batch_size: int = 1000,
) -> Dataset:
    """Cast the given feature columns (and nested sequences) to float32.

    Model feature/extracted columns must be float for concatenation in the model.
    Operates on a single ``Dataset`` and returns the cast dataset.
    """

    def cast_to_float(feature):
        if isinstance(feature, Sequence):
            return Sequence(cast_to_float(feature.feature))
        if isinstance(feature, Value):
            return Value("float32")
        return feature

    feature_columns = set(feature_columns)
    if not feature_columns:
        return dataset

    new_features = dataset.features.copy()
    for name, ftype in dataset.features.items():
        if name in feature_columns:
            new_features[name] = cast_to_float(ftype)

    return dataset.cast(new_features, num_proc=num_proc, batch_size=batch_size)


def to_tf_tensor_dataset(
    hf_dataset: Dataset,
    input_columns: List[str],
    batch_size: int,
    label_cols: Optional[Union[str, List[str]]] = None,
    shuffle: bool = False,
):
    """Build a ``tf.data.Dataset`` from a processed HF dataset.

    When ``label_cols`` is None, only input tensors are produced (inference).
    A single-element label list is collapsed to a scalar column for API stability.
    """
    kwargs = {"columns": input_columns, "shuffle": shuffle, "batch_size": batch_size}

    if label_cols is not None:
        if isinstance(label_cols, list) and len(label_cols) == 1:
            label_cols = label_cols[0]
        kwargs["label_cols"] = label_cols

    return hf_dataset.to_tf_dataset(**kwargs)


def to_torch_dataloader(
    hf_dataset: Dataset,
    input_columns: List[str],
    batch_size: int,
    label_cols: Optional[List[str]] = None,
    shuffle: bool = False,
    dataloader_kwargs: Optional[dict] = None,
):
    """Build a torch ``DataLoader`` from a processed HF dataset.

    When ``label_cols`` is None, only input columns are formatted (inference).
    ``dataloader_kwargs`` (excluding ``dataset``) override the defaults.
    """
    from torch.utils.data import DataLoader

    columns = list(input_columns)
    if label_cols:
        columns = [*columns, *label_cols]

    kwargs = {
        "dataset": hf_dataset.with_format(type="torch", columns=columns),
        "batch_size": batch_size,
        "shuffle": shuffle,
    }

    if dataloader_kwargs:
        kwargs.update({k: v for k, v in dataloader_kwargs.items() if k != "dataset"})

    return DataLoader(**kwargs)
