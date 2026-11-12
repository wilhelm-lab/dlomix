"""
Shared helpers to convert a processed HuggingFace ``Dataset`` into backend tensors.

These are used both by :class:`~dlomix.data.dataset.PeptideDataset` (per split, with
labels) and by :class:`~dlomix.data.inference.PeptidePreprocessor` (inference, no
labels). Kept free of dlomix-internal imports to avoid circular dependencies.
"""

from typing import Iterable, List, Optional, Union

import numpy as np
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


def _tf_numpy_dtype(dtype: np.dtype) -> type:
    """The NumPy type a column is fed to TensorFlow as: int64, float32 or str."""
    if np.issubdtype(dtype, np.integer) or dtype == bool:
        return np.int64
    if np.issubdtype(dtype, np.number):
        return np.float32
    if dtype.kind == "U":
        return np.str_
    raise TypeError(
        f"Columns of dtype {dtype} cannot be converted to TensorFlow tensors."
    )


def load_tf_columns_in_memory(hf_dataset: Dataset, columns: List[str]) -> dict:
    """Load whole columns of a processed HF dataset as TensorFlow tensors.

    The tensors live in host memory (never on a GPU) and need about as much RAM as
    the columns take on disk. Pass the result to :func:`to_tf_tensor_dataset` as
    ``in_memory_columns``.
    """
    import tensorflow as tf

    arrays = hf_dataset.with_format("numpy", columns=columns)[:]
    with tf.device("CPU:0"):
        return {
            column: tf.constant(
                np.asarray(arrays[column], dtype=_tf_numpy_dtype(arrays[column].dtype))
            )
            for column in columns
        }


def _arrow_batch_reader(hf_dataset: Dataset, columns: List[str]):
    """A ``tf.data`` map function reading a batch of rows, given their indices, from
    the dataset's Arrow file. It runs in Python, one batch at a time."""
    import tensorflow as tf

    table = hf_dataset.with_format("numpy", columns=columns)
    sample = table[:1]
    dtypes = {column: _tf_numpy_dtype(sample[column].dtype) for column in columns}
    shapes = {column: (None, *sample[column].shape[1:]) for column in columns}

    def read(indices):
        # consecutive rows (unshuffled data) are read as a slice, which is faster
        if len(indices) > 0 and np.all(np.diff(indices) == 1):
            batch = table[int(indices[0]) : int(indices[-1]) + 1]
        else:
            batch = table[indices]
        return [np.asarray(batch[column], dtype=dtypes[column]) for column in columns]

    def fetch(indices):
        values = tf.numpy_function(
            read, [indices], [tf.as_dtype(dtypes[column]) for column in columns]
        )
        return {
            column: tf.ensure_shape(value, shapes[column])
            for column, value in zip(columns, values)
        }

    return fetch


def to_tf_tensor_dataset(
    hf_dataset: Dataset,
    input_columns: List[str],
    batch_size: int,
    label_cols: Optional[Union[str, List[str]]] = None,
    shuffle: bool = False,
    in_memory_columns: Optional[dict] = None,
):
    """Build a ``tf.data.Dataset`` from a processed HF dataset.

    Batches are built from a row index: ``range(n)``, shuffled if ``shuffle``,
    batched, then mapped to the rows' tensors. With ``shuffle``, every epoch (every
    iteration of the dataset) visits every row once in a new order, also when a
    global seed is set (``keras.utils.set_random_seed``), which makes the sequence of
    orders reproducible. Without it, batches follow the dataset's row order.

    By default each batch is read from the dataset's Arrow file. With
    ``in_memory_columns`` (see :func:`load_tf_columns_in_memory`) the batches are
    gathered from these tensors instead, which is much faster and yields the same
    batches.

    When ``label_cols`` is None, only input tensors are produced (inference).
    Several input (or label) columns give a dict of tensors, a single one its tensor.
    """
    import tensorflow as tf

    if isinstance(label_cols, str):
        label_cols = [label_cols]
    label_cols = list(label_cols or [])
    columns = [*input_columns, *label_cols]

    if in_memory_columns is not None:
        tensors = {column: in_memory_columns[column] for column in columns}

        def fetch(indices):
            return {column: tf.gather(tensors[column], indices) for column in columns}

        num_parallel_calls = tf.data.AUTOTUNE
    else:
        fetch = _arrow_batch_reader(hf_dataset, columns)
        num_parallel_calls = None

    def select(batch, names):
        if len(names) == 1:
            return batch[names[0]]
        return {name: batch[name] for name in names}

    def structure(batch):
        if not label_cols:
            return select(batch, input_columns)
        return select(batch, input_columns), select(batch, label_cols)

    n_rows = len(hf_dataset)
    rows = tf.data.Dataset.range(n_rows)
    if shuffle and n_rows > 1:
        rows = rows.shuffle(n_rows, reshuffle_each_iteration=True)

    return (
        rows.batch(batch_size)
        .map(lambda indices: structure(fetch(indices)), num_parallel_calls)
        .prefetch(tf.data.AUTOTUNE)
    )


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
