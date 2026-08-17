"""Tests for PeptidePreprocessor (reusable inference preprocessing)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from dlomix.config import _BACKEND, PYTORCH_BACKEND
from dlomix.data import PeptidePreprocessor, RetentionTimeDataset

DATASET_TYPE = "pt" if _BACKEND in PYTORCH_BACKEND else "tf"

RAW_SEQUENCES = [
    "ACDEFGHIK",
    "PEPTIDEK",
    "MKLVAAR",
    "GGGGSSSK",
    "ACDEFGHIKLMN",
    "PEPK",
    "MMMKL",
    "AAACCCDDD",
    "KKLLMMNN",
    "PPQQRRSS",
]


@pytest.fixture
def rt_dataset():
    seqs = RAW_SEQUENCES * 4
    data = {
        "modified_sequence": seqs,
        "indexed_retention_time": [float(len(s)) for s in seqs],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RetentionTimeDataset(
            data_source=Dataset.from_dict(data),
            data_format="hf",
            sequence_column="modified_sequence",
            label_column="indexed_retention_time",
            val_ratio=0.2,
            max_seq_len=20,
            batch_size=8,
            dataset_type=DATASET_TYPE,
        )


def _first_batch_seq(tensors, seq_col):
    """Extract the first batch's sequence tensor as numpy, for tf or torch outputs."""
    batch = next(iter(tensors))
    arr = batch[seq_col] if isinstance(batch, dict) else batch
    return np.asarray(arr)


def test_get_preprocessor_returns_preprocessor(rt_dataset):
    prep = rt_dataset.get_preprocessor()
    assert isinstance(prep, PeptidePreprocessor)
    assert prep.sequence_column == "modified_sequence"
    assert prep.max_seq_len == 20
    assert prep.vocab_size == len(rt_dataset.extended_alphabet)
    assert prep.dataset_type == DATASET_TYPE


def test_transform_shape(rt_dataset):
    prep = rt_dataset.get_preprocessor()
    seqs = ["ACDEFGHIK", "PEPTIDEK"]
    out = prep(seqs)
    arr = _first_batch_seq(out, prep.sequence_column)
    # with_termini -> max_seq_len + 2 columns
    assert arr.shape == (2, 22)


def test_single_string_input(rt_dataset):
    prep = rt_dataset.get_preprocessor()
    arr = _first_batch_seq(prep("ACDEFGHIK"), prep.sequence_column)
    assert arr.shape == (1, 22)


def test_input_formats_equivalent(rt_dataset):
    prep = rt_dataset.get_preprocessor()
    seqs = ["ACDEFGHIK", "PEPTIDEK"]

    from_list = _first_batch_seq(prep(seqs), prep.sequence_column)
    from_np = _first_batch_seq(prep(np.array(seqs)), prep.sequence_column)
    from_dict = _first_batch_seq(
        prep({prep.sequence_column: seqs}), prep.sequence_column
    )
    from_df = _first_batch_seq(
        prep(pd.DataFrame({prep.sequence_column: seqs})), prep.sequence_column
    )
    from_hf = _first_batch_seq(
        prep(Dataset.from_dict({prep.sequence_column: seqs})), prep.sequence_column
    )

    for other in (from_np, from_dict, from_df, from_hf):
        np.testing.assert_array_equal(from_list, other)


def test_encoding_matches_alphabet(rt_dataset):
    """The encoded sequence uses the dataset's learned alphabet + padding."""
    prep = rt_dataset.get_preprocessor()
    alphabet = rt_dataset.extended_alphabet
    arr = _first_batch_seq(prep(["ACDEK"]), prep.sequence_column)

    expected_tokens = ["[]-", "A", "C", "D", "E", "K", "-[]"]
    expected_ids = [alphabet[t] for t in expected_tokens]
    pad_id = alphabet[prep.padding_value]

    assert list(arr[0, : len(expected_ids)]) == expected_ids
    assert set(arr[0, len(expected_ids) :]) == {pad_id}


def test_missing_sequence_column_raises(rt_dataset):
    prep = rt_dataset.get_preprocessor()
    with pytest.raises(ValueError, match="Sequence column.*not found"):
        prep({"wrong_column": ["ACDEK"]})


def test_save_load_roundtrip(rt_dataset, tmp_path):
    prep = rt_dataset.get_preprocessor()
    artifact = prep.save(str(tmp_path / "prep"))

    loaded = PeptidePreprocessor.load(artifact)
    assert loaded.fingerprint == prep.fingerprint
    assert loaded.alphabet == prep.alphabet
    assert loaded.max_seq_len == prep.max_seq_len

    a = _first_batch_seq(prep(["PEPTIDEK"]), prep.sequence_column)
    b = _first_batch_seq(loaded(["PEPTIDEK"]), loaded.sequence_column)
    np.testing.assert_array_equal(a, b)


def test_from_saved_dataset_dir(rt_dataset, tmp_path):
    save_dir = str(tmp_path / "dataset")
    rt_dataset.save_to_disk(save_dir)

    prep = PeptidePreprocessor.from_saved(save_dir)
    assert prep.vocab_size == len(rt_dataset.extended_alphabet)

    live = rt_dataset.get_preprocessor()
    a = _first_batch_seq(prep(["PEPTIDEK"]), prep.sequence_column)
    b = _first_batch_seq(live(["PEPTIDEK"]), live.sequence_column)
    np.testing.assert_array_equal(a, b)


def test_manual_construction_requires_padding_in_alphabet():
    with pytest.raises(ValueError, match="padding_value.*not present"):
        PeptidePreprocessor(
            alphabet={"A": 2, "C": 3},  # no padding token
            sequence_column="seq",
            max_seq_len=10,
            dataset_type=DATASET_TYPE,
        )
