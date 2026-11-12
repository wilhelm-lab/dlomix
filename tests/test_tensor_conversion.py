import numpy as np
import pytest
from datasets import Dataset

from dlomix.data import RetentionTimeDataset
from dlomix.data.tensor_conversion import (
    load_tf_columns_in_memory,
    to_tf_tensor_dataset,
)

tf = pytest.importorskip("tensorflow")

N_ROWS, BATCH_SIZE = 23, 5  # the last batch is partial
INPUTS, LABEL = ["sequence", "collision_energy", "method"], "intensities"


@pytest.fixture(scope="module")
def hf_dataset():
    rows = range(N_ROWS)
    return Dataset.from_dict(
        {
            "sequence": [[row] * 4 for row in rows],
            "collision_energy": [row / 10 for row in rows],
            "method": [f"m{row}" for row in rows],
            "intensities": [[float(row)] * 3 for row in rows],
        }
    )


def _tensor_dataset(hf_dataset, in_memory, **kwargs):
    columns = load_tf_columns_in_memory(hf_dataset, [*INPUTS, LABEL])
    return to_tf_tensor_dataset(
        hf_dataset,
        INPUTS,
        BATCH_SIZE,
        label_cols=[LABEL],
        in_memory_columns=columns if in_memory else None,
        **kwargs,
    )


def _row_order(tensor_dataset, epochs):
    """The rows each epoch yields, read back from the sequence column."""
    return [
        [int(row) for inputs, _ in tensor_dataset for row in inputs["sequence"][:, 0]]
        for _ in range(epochs)
    ]


@pytest.mark.parametrize("in_memory", [False, True])
def test_unshuffled_batches_equal_hf_to_tf_dataset(hf_dataset, in_memory):
    expected = hf_dataset.to_tf_dataset(
        columns=INPUTS, label_cols=LABEL, batch_size=BATCH_SIZE
    )
    batches = _tensor_dataset(hf_dataset, in_memory)

    assert batches.element_spec == expected.element_spec
    for (inputs, labels), (expected_inputs, expected_labels) in zip(
        batches, expected, strict=True
    ):
        for column in INPUTS:
            np.testing.assert_array_equal(inputs[column], expected_inputs[column])
        np.testing.assert_array_equal(labels, expected_labels)


@pytest.mark.parametrize("in_memory", [False, True])
def test_shuffled_epochs_differ_and_visit_every_row_once(hf_dataset, in_memory):
    tf.keras.utils.set_random_seed(0)
    first, second = _row_order(
        _tensor_dataset(hf_dataset, in_memory, shuffle=True), epochs=2
    )

    assert first != second
    assert sorted(first) == sorted(second) == list(range(N_ROWS))


def test_shuffled_batches_depend_only_on_the_seed(hf_dataset):
    def epochs(seed, in_memory):
        tf.keras.utils.set_random_seed(seed)
        return _row_order(_tensor_dataset(hf_dataset, in_memory, shuffle=True), 2)

    from_disk = epochs(seed=0, in_memory=False)
    assert epochs(seed=0, in_memory=False) == from_disk
    assert epochs(seed=0, in_memory=True) == from_disk
    assert epochs(seed=1, in_memory=False) != from_disk


def test_inference_dataset_has_inputs_only(hf_dataset):
    batch = next(iter(to_tf_tensor_dataset(hf_dataset, ["sequence"], BATCH_SIZE)))
    np.testing.assert_array_equal(batch, [[row] * 4 for row in range(BATCH_SIZE)])


def test_dataset_flag_keeps_one_copy_of_a_split_in_memory():
    dataset = RetentionTimeDataset(
        data_source=Dataset.from_dict(
            {"sequence": ["AAK", "CDE", "FGH", "IKL"], "irt": [1.0, 2, 3, 4]}
        ),
        data_format="hf",
        sequence_column="sequence",
        label_column="irt",
        max_seq_len=5,
        batch_size=2,
        val_ratio=0.5,
        dataset_type="tf",
    )

    def batches(split, enable_tf_dataset_cache):
        dataset.enable_tf_dataset_cache = enable_tf_dataset_cache
        tf.keras.utils.set_random_seed(0)
        return list(getattr(dataset, split).as_numpy_iterator())

    for split in ("tensor_train_data", "tensor_val_data"):
        np.testing.assert_equal(batches(split, True), batches(split, False))

    # asking for the tensor dataset again does not load the split again
    loaded = dataset._in_memory_splits["train"][1]
    dataset.enable_tf_dataset_cache = True
    dataset.tensor_train_data
    assert dataset._in_memory_splits["train"][1] is loaded
