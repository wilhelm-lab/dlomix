import logging
import time
from os.path import join
from shutil import rmtree

import pytest
import torch
from datasets import Dataset, DatasetDict, load_dataset

from dlomix.constants import ALPHABET_UNMOD
from dlomix.data import (
    FragmentIonIntensityDataset,
    IonMobilityDataset,
    RetentionTimeDataset,
    load_processed_dataset,
)
from dlomix.data.dataset_utils import EncodingScheme

logger = logging.getLogger(__name__)

RT_HUB_DATASET_NAME = "Wilhelmlab/prospect-ptms-irt"


def test_empty_rtdataset():
    rtdataset = RetentionTimeDataset()
    assert rtdataset.hf_dataset is None
    assert rtdataset._empty_dataset_mode is True


def test_num_proc_minus_one_uses_available_processors(monkeypatch):
    monkeypatch.setattr("dlomix.data.dataset.get_num_processors", lambda: 6)

    dataset = RetentionTimeDataset(num_proc=-1)

    assert dataset._num_proc == 6


def test_num_proc_none_forces_single_process(monkeypatch):
    monkeypatch.setattr("dlomix.data.dataset.get_num_processors", lambda: 6)

    dataset = RetentionTimeDataset(num_proc=None)

    assert dataset._num_proc is None


def test_num_proc_user_value_is_capped_to_available(monkeypatch):
    monkeypatch.setattr("dlomix.data.dataset.get_num_processors", lambda: 6)

    dataset = RetentionTimeDataset(num_proc=10)

    assert dataset._num_proc == 6


def test_parquet_rtdataset(download_path_for_assets):
    rtdataset = RetentionTimeDataset(
        data_source=join(download_path_for_assets, "file_1.parquet"),
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        val_ratio=0.2,
    )
    assert rtdataset.hf_dataset is not None
    assert rtdataset._empty_dataset_mode is False
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[0] in list(
        rtdataset.hf_dataset.keys()
    )
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[1] in list(
        rtdataset.hf_dataset.keys()
    )
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[2] not in list(
        rtdataset.hf_dataset.keys()
    )
    assert rtdataset[RetentionTimeDataset.DEFAULT_SPLIT_NAMES[0]].num_rows > 0
    assert rtdataset[RetentionTimeDataset.DEFAULT_SPLIT_NAMES[1]].num_rows > 0


def test_rtdataset_inmemory(download_path_for_assets):
    hf_dataset = load_dataset(
        "parquet",
        data_files=join(download_path_for_assets, "file_1.parquet"),
        split="train",
    )

    rtdataset = RetentionTimeDataset(
        data_source=hf_dataset,
        data_format="hf",
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        val_ratio=0.2,
    )
    assert rtdataset.hf_dataset is not None
    assert rtdataset._empty_dataset_mode is False
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[0] in list(
        rtdataset.hf_dataset.keys()
    )

    assert rtdataset[RetentionTimeDataset.DEFAULT_SPLIT_NAMES[0]].num_rows > 0


def test_rtdataset_hub():
    rtdataset = RetentionTimeDataset(
        data_source=RT_HUB_DATASET_NAME,
        data_format="hub",
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        name="holdout",
        # the holdout config has only a test split, so there is nothing to learn
        # an alphabet from
        alphabet=ALPHABET_UNMOD,
    )
    logger.info(rtdataset)
    assert rtdataset.hf_dataset is not None
    assert rtdataset._empty_dataset_mode is False

    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[2] in list(
        rtdataset.hf_dataset.keys()
    )

    assert rtdataset[RetentionTimeDataset.DEFAULT_SPLIT_NAMES[2]].num_rows > 0


def test_csv_rtdataset(download_path_for_assets):
    rtdataset = RetentionTimeDataset(
        data_source=join(download_path_for_assets, "file_2.csv"),
        data_format="csv",
        sequence_column="sequence",
        label_column="irt",
        val_ratio=0.2,
    )

    assert rtdataset.hf_dataset is not None
    assert rtdataset._empty_dataset_mode is False
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[0] in list(
        rtdataset.hf_dataset.keys()
    )
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[1] in list(
        rtdataset.hf_dataset.keys()
    )
    assert RetentionTimeDataset.DEFAULT_SPLIT_NAMES[2] not in list(
        rtdataset.hf_dataset.keys()
    )
    assert rtdataset[RetentionTimeDataset.DEFAULT_SPLIT_NAMES[0]].num_rows > 0
    assert rtdataset[RetentionTimeDataset.DEFAULT_SPLIT_NAMES[1]].num_rows > 0


def test_empty_intensitydataset():
    intensity_dataset = FragmentIonIntensityDataset()
    assert intensity_dataset.hf_dataset is None
    assert intensity_dataset._empty_dataset_mode is True


def test_parquet_intensitydataset(download_path_for_assets):
    filepath = join(download_path_for_assets, "file_3.parquet")
    intensity_dataset = FragmentIonIntensityDataset(
        data_format="parquet",
        data_source=filepath,
        sequence_column="sequence",
        label_column="intensities",
        model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
        val_ratio=0.2,
    )

    assert intensity_dataset.hf_dataset is not None
    assert intensity_dataset._empty_dataset_mode is False
    assert FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[0] in list(
        intensity_dataset.hf_dataset.keys()
    )
    assert FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[1] in list(
        intensity_dataset.hf_dataset.keys()
    )
    assert FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[2] not in list(
        intensity_dataset.hf_dataset.keys()
    )
    assert (
        intensity_dataset[FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[0]].num_rows
        > 0
    )
    assert (
        intensity_dataset[FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[1]].num_rows
        > 0
    )


def test_nested_model_features(raw_generic_nested_data):
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
        val_ratio=0.5,
    )

    assert intensity_dataset.hf_dataset is not None
    assert intensity_dataset._empty_dataset_mode is False

    example = iter(intensity_dataset.tensor_train_data).next()
    assert example[0]["nested_feature"].shape == [1, 1, 2]


def test_save_dataset(raw_generic_nested_data):
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
        val_ratio=0.1,
    )

    save_path = "./.test_dataset_2"
    attributes = intensity_dataset.__dict__
    logger.info("Dataset attributes before saving: {}".format(attributes))

    intensity_dataset.save_to_disk(save_path, overwrite=True)
    rmtree(save_path)


def test_load_dataset(download_path_for_assets):
    rtdataset = RetentionTimeDataset(
        data_source=join(download_path_for_assets, "file_2.csv"),
        data_format="csv",
        sequence_column="sequence",
        label_column="irt",
        val_ratio=0.2,
    )

    save_path = "./.test_dataset_1"
    rtdataset.save_to_disk(save_path, overwrite=True)
    splits = rtdataset._data_files_available_splits
    config = rtdataset._config

    load_time_threshold = 0.05  # 50ms

    start_time = time.time()
    loaded_dataset = load_processed_dataset(save_path)
    load_duration = time.time() - start_time
    logger.info("Loaded the dataset in {} seconds".format(load_duration))

    logger.info("Original datasets config: {}".format(rtdataset._config))
    logger.info("Loaded datasets config: {}".format(loaded_dataset._config))

    # Assert the load time is below the threshold
    assert (
        load_duration < load_time_threshold
    ), f"Load time exceeded: {load_duration:.3f}s"
    assert loaded_dataset.processed is True

    assert loaded_dataset._data_files_available_splits == splits
    assert loaded_dataset.hf_dataset is not None
    assert loaded_dataset._config == config, f"{loaded_dataset._config} != {config}"
    rmtree(save_path)


def test_no_split_datasetDict_hf_inmemory(raw_generic_nested_data):
    hfdata = Dataset.from_dict(raw_generic_nested_data)
    hf_dataset = DatasetDict({"train": hfdata})

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hf_dataset,
        sequence_column="seq",
        label_column="label",
    )

    assert intensity_dataset.hf_dataset is not None
    assert intensity_dataset._empty_dataset_mode is False
    assert FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[0] in list(
        intensity_dataset.hf_dataset.keys()
    )

    assert (
        len(
            intensity_dataset.hf_dataset[
                FragmentIonIntensityDataset.DEFAULT_SPLIT_NAMES[0]
            ]
        )
        == 2
    )

    # test learning alphabet for train/val and then using it for test with fallback


def _make_rt_split_data(seqs):
    return Dataset.from_dict(
        {
            "modified_sequence": seqs,
            "indexed_retention_time": [0.1 + i for i in range(len(seqs))],
        }
    )


def test_encoding_learning_forces_single_process(monkeypatch):
    # Keep split insertion order as train -> test -> val to capture ordering assumptions.
    hf_dataset = DatasetDict(
        {
            "train": _make_rt_split_data(["[]-AC-[]"]),
            "test": _make_rt_split_data(["[]-C[UNIMOD:4]A-[]"]),
            "val": _make_rt_split_data(["[]-C[UNIMOD:4]A-[]"]),
        }
    )

    calls = []
    original_map = Dataset.map

    def map_spy(self, function, *args, **kwargs):
        calls.append((kwargs.get("desc"), kwargs.get("num_proc")))
        return original_map(self, function, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", map_spy)

    RetentionTimeDataset(
        data_format="hf",
        data_source=hf_dataset,
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        encoding_scheme=EncodingScheme.NAIVE_MODS,
        alphabet=None,
        num_proc=2,
        max_seq_len=8,
    )

    encoding_calls = [
        c for c in calls if c[0].startswith("Mapping SequenceEncodingProcessor")
    ]
    assert len(encoding_calls) == 3

    # Encoding is deterministic train -> val -> test.
    # train/val must force single-process learning, test keeps configured num_proc.
    assert encoding_calls[0][1] is None
    assert encoding_calls[1][1] is None
    assert encoding_calls[2][1] == 2


def test_val_tokens_available_to_test_even_with_nonstandard_split_order():
    # Token appears in val and test, but not train. If test is encoded before val,
    # fallback may be used incorrectly instead of learned token encoding.
    hf_dataset = DatasetDict(
        {
            "train": _make_rt_split_data(["[]-AC-[]"]),
            "test": _make_rt_split_data(["[]-C[UNIMOD:4]A-[]"]),
            "val": _make_rt_split_data(["[]-C[UNIMOD:4]A-[]"]),
        }
    )

    dataset = RetentionTimeDataset(
        data_format="hf",
        data_source=hf_dataset,
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        encoding_scheme=EncodingScheme.NAIVE_MODS,
        alphabet=None,
        num_proc=2,
        max_seq_len=8,
    )

    assert "C[UNIMOD:4]" in dataset.extended_alphabet

    test_encoded = dataset.hf_dataset["test"][0]["modified_sequence"]
    learned_token_index = dataset.extended_alphabet["C[UNIMOD:4]"]

    # with_termini=True means sequence starts with []- at index 0.
    assert test_encoded[1] == learned_token_index


def test_ionmobility_dataset_inmemory():
    data = Dataset.from_dict(
        {
            "sequence_modified": ["ACDEK", "PEPTIDEK", "MKLVAAR", "GGSSK"] * 5,
            "ccs": [300.0 + i for i in range(20)],
            "ccs_std": [2.0] * 20,
            "charge": [2, 3, 2, 1] * 5,
            "mz": [500.0 + i for i in range(20)],
        }
    )

    dataset = IonMobilityDataset(
        data_format="hf",
        data_source=data,
        val_ratio=0.2,
        max_seq_len=10,
        batch_size=4,
        shuffle=True,
    )

    assert set(dataset.hf_dataset.keys()) == {"train", "val"}
    assert dataset.shuffle is True
    batch = next(iter(dataset.tensor_train_data))
    assert batch is not None


@pytest.mark.parametrize("source", ["hf_test_split", "test_data_source_file"])
def test_test_only_dataset_requires_alphabet(source, tmp_path):
    # The alphabet is learned on train/val only; a test-only dataset without one
    # would silently encode every residue as the unknown token.
    test_split = _make_rt_split_data(["ACDEK", "PEPTIDEK"])
    if source == "hf_test_split":
        source_kwargs = {
            "data_format": "hf",
            "data_source": DatasetDict({"test": test_split}),
        }
    else:
        test_file = str(tmp_path / "test.csv")
        test_split.to_csv(test_file)
        source_kwargs = {"data_format": "csv", "test_data_source": test_file}

    rt_kwargs = dict(
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        max_seq_len=10,
        **source_kwargs,
    )

    with pytest.raises(ValueError, match="alphabet is required"):
        RetentionTimeDataset(**rt_kwargs)

    train = RetentionTimeDataset(
        data_format="hf",
        data_source=DatasetDict({"train": _make_rt_split_data(["ACDEKPTI"])}),
        sequence_column="modified_sequence",
        label_column="indexed_retention_time",
        max_seq_len=10,
    )
    test = RetentionTimeDataset(alphabet=train.extended_alphabet, **rt_kwargs)

    # every residue of the first test peptide is encoded with the training alphabet
    encoded = test.hf_dataset["test"][0]["modified_sequence"]
    assert encoded[1:6] == [train.extended_alphabet[aa] for aa in "ACDEK"]


def test_shuffle_parameter(raw_generic_nested_data):
    """Test that shuffle parameter works for both TensorFlow and PyTorch datasets."""
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    # Test with shuffle=True for TensorFlow
    tf_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        dataset_type="tf",
        shuffle=True,
        batch_size=1,
        val_ratio=0.2,
    )

    # Test with shuffle=True for PyTorch
    pt_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        dataset_type="pt",
        shuffle=True,
        batch_size=1,
        val_ratio=0.2,
    )

    # Verify datasets are created successfully
    assert tf_dataset.shuffle is True
    assert pt_dataset.shuffle is True
    assert tf_dataset.tensor_train_data is not None
    assert pt_dataset.tensor_train_data is not None


def test_torch_dataloader_kwargs(raw_generic_nested_data):
    """Test that additional PyTorch DataLoader kwargs are properly passed through."""
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        dataset_type="pt",
        batch_size=1,
        val_ratio=0.2,
        torch_dataloader_kwargs={
            "drop_last": True,
            "pin_memory": False,
            "num_workers": 0,  # Use 0 to avoid multiprocessing issues in tests
        },
    )

    # Get the DataLoader
    dataloader = dataset.tensor_train_data

    # Verify that torch_dataloader_kwargs were applied
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is False
    assert dataloader.num_workers == 0
    assert dataset.torch_dataloader_kwargs is not None


def test_dataset_torch(raw_generic_nested_data):
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
        dataset_type="pt",
        batch_size=2,
        max_seq_len=15,
        with_termini=False,
        val_ratio=0.5,
    )

    logger.info(intensity_dataset)
    assert intensity_dataset.hf_dataset is not None
    assert intensity_dataset._empty_dataset_mode is False

    batch = next(iter(intensity_dataset.tensor_train_data))

    logger.info(batch)

    assert list(batch["nested_feature"].shape) == [1, 1, 2]
    assert list(batch["seq"].shape) == [1, 15]
    assert list(batch["label"].shape) == [
        1,
    ]

    assert batch["seq"].dtype == torch.int64
    assert batch["label"].dtype == torch.float32


def test_tf_tensor_dataset_string_label(raw_generic_nested_data):
    """Test that TensorFlow TensorDataset is created properly."""
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        dataset_type="tf",
        batch_size=1,
        val_ratio=0.2,
    )

    # Get the TensorFlow dataset
    tf_dataset = dataset.tensor_train_data

    # Verify that the TensorFlow dataset is created successfully
    assert tf_dataset is not None
    for batch in tf_dataset.take(1):
        features, labels = batch
        assert features is not None
        assert labels is not None


def test_tf_tensor_dataset_singelton_list_label(raw_generic_nested_data):
    """Test that TensorFlow TensorDataset is created properly."""
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column=["label"],
        dataset_type="tf",
        batch_size=1,
        val_ratio=0.2,
    )

    # Get the TensorFlow dataset
    tf_dataset = dataset.tensor_train_data

    # Verify that the TensorFlow dataset is created successfully
    assert tf_dataset is not None
    for batch in tf_dataset.take(1):
        features, labels = batch
        assert features is not None
        assert labels is not None


def test_tf_tensor_dataset_list_multi_label(raw_generic_nested_data):
    """Test that TensorFlow TensorDataset is created properly."""
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column=["label", "label2"],
        dataset_type="tf",
        batch_size=1,
        val_ratio=0.2,
    )

    # Get the TensorFlow dataset
    tf_dataset = dataset.tensor_train_data

    # Verify that the TensorFlow dataset is created successfully
    assert tf_dataset is not None
    for batch in tf_dataset.take(1):
        features, labels = batch
        assert features is not None
        assert labels is not None


# Integration tests for dataset splitter
def test_rtdataset_with_sequence_unique_splitter(download_path_for_assets):
    """Test RetentionTimeDataset with sequence-unique splitting."""
    # Use a dataset from HF directly to avoid the processing step modifying sequences
    from datasets import load_dataset as hf_load_dataset

    hf_data = hf_load_dataset(
        "csv",
        data_files=join(download_path_for_assets, "file_2.csv"),
        split="train",
    )

    rtdataset = RetentionTimeDataset(
        data_source=hf_data,
        data_format="hf",
        sequence_column="sequence",
        label_column="irt",
        val_ratio=0.2,
        split_strategy="sequence_unique",
        split_seed=42,
    )

    assert rtdataset.hf_dataset is not None

    # Note: After processing, sequence column is still present but may be modified
    # We verify the split happened correctly by checking dataset existence
    assert "train" in rtdataset.hf_dataset
    assert "val" in rtdataset.hf_dataset
    assert rtdataset["train"].num_rows > 0
    assert rtdataset["val"].num_rows > 0


def test_rtdataset_split_config_conflict(download_path_for_assets):
    """Test that providing both predefined splits and split config raises error."""
    with pytest.raises(ValueError, match="Cannot use split configuration parameters"):
        RetentionTimeDataset(
            data_source=join(download_path_for_assets, "file_2.csv"),
            val_data_source=join(download_path_for_assets, "file_2.csv"),
            data_format="csv",
            sequence_column="sequence",
            label_column="irt",
            split_strategy="sequence_unique",  # Conflict: predefined val_data_source + split_strategy
            split_seed=42,
        )


def _private_cache_source(download_path_for_assets, tmp_path):
    """file_3.parquet as a dataset whose cache files live in a private folder, so the
    cache tests neither hit files left by other tests nor collide with other runs."""
    return load_dataset(
        "parquet",
        data_files=join(download_path_for_assets, "file_3.parquet"),
        cache_dir=str(tmp_path / "hf_cache"),
    )["train"]


def test_learned_alphabet_survives_cached_encoding(download_path_for_assets, tmp_path):
    # The alphabet is learned as a side effect of the encoding map. When that map's
    # result was cached by an earlier build (auto_cleanup_cache=False, or a crashed
    # run), reusing the cache skipped the learning: the train split kept the first
    # build's encoding while the alphabet was learned from val alone.
    kwargs = dict(
        data_source=_private_cache_source(download_path_for_assets, tmp_path),
        data_format="hf",
        sequence_column="sequence",
        label_column="intensities",
        model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
        val_ratio=0.1,
        split_seed=1,
        num_proc=None,
        auto_cleanup_cache=False,
    )
    first = FragmentIonIntensityDataset(**kwargs)
    second = FragmentIonIntensityDataset(**kwargs)
    assert second.extended_alphabet == first.extended_alphabet
    assert (
        second.hf_dataset["train"]["sequence"] == first.hf_dataset["train"]["sequence"]
    )


def test_tokens_missing_from_an_explicit_alphabet_are_reported():
    # train/val encode them as X, test falls back to the unmodified residue; both
    # used to happen silently
    data = Dataset.from_dict(
        {
            "modified_sequence": [
                "[]-PEPM[UNIMOD:35]K-[]",
                "[]-AC[UNIMOD:4]DK-[]",
                "[]-PEPS[UNIMOD:21]K-[]",
            ],
            "indexed_retention_time": [1.0, 2.0, 3.0],
        }
    )
    alphabet = {
        **{aa: i for i, aa in enumerate("-XPEMKACDS")},
        "C[UNIMOD:4]": 10,
        "[]-": 11,
        "-[]": 12,
    }
    with pytest.warns(UserWarning, match="Tokens missing from the alphabet") as record:
        RetentionTimeDataset(
            data_source=DatasetDict({"train": data, "test": data}),
            data_format="hf",
            alphabet=alphabet,
            encoding_scheme="naive-mods",
            num_proc=None,
        )
    message = next(str(w.message) for w in record if "Tokens missing" in str(w.message))
    assert "train: 2 occurrences of 2 tokens" in message
    assert "test: 2 occurrences of 2 tokens" in message
    assert "M[UNIMOD:35]" in message and "S[UNIMOD:21]" in message
    assert "C[UNIMOD:4]" not in message


def test_learned_alphabet_reports_only_unseen_test_tokens():
    train = Dataset.from_dict(
        {"modified_sequence": ["[]-PEPK-[]"] * 2, "indexed_retention_time": [1.0, 2.0]}
    )
    test = Dataset.from_dict(
        {
            "modified_sequence": ["[]-PEPM[UNIMOD:35]K-[]"],
            "indexed_retention_time": [1.0],
        }
    )
    with pytest.warns(UserWarning, match="Tokens missing") as record:
        RetentionTimeDataset(
            data_source=DatasetDict({"train": train, "test": test}),
            data_format="hf",
            encoding_scheme="naive-mods",
            num_proc=None,
        )
    message = next(str(w.message) for w in record if "Tokens missing" in str(w.message))
    assert "train:" not in message
    assert "test: 1 occurrences of 1 tokens" in message


def test_with_termini_false_warns_about_dropped_terminal_mods():
    data = Dataset.from_dict(
        {
            "modified_sequence": [
                "[UNIMOD:737]-PEPK-[]",
                "[UNIMOD:737]-ACDK-[]",
                "[UNIMOD:1]-PEPK-[]",
                "[]-PEPK-[]",
            ],
            "indexed_retention_time": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.warns(UserWarning, match="drops terminal modifications") as record:
        RetentionTimeDataset(
            data_source=DatasetDict({"train": data}),
            data_format="hf",
            encoding_scheme="naive-mods",
            with_termini=False,
            num_proc=None,
        )
    message = next(str(w.message) for w in record if "drops terminal" in str(w.message))
    assert "3 sequences" in message
    assert "'[UNIMOD:737]-': 2" in message and "'[UNIMOD:1]-': 1" in message


def test_dataset_columns_to_keep_is_not_modified():
    # the label is kept anyway; naming it again must not duplicate the column
    keep = ["indexed_retention_time"]
    data = Dataset.from_dict(
        {
            "modified_sequence": ["[]-PEPK-[]", "[]-ACDK-[]"],
            "indexed_retention_time": [1.0, 2.0],
        }
    )
    RetentionTimeDataset(
        data_source=DatasetDict({"train": data}),
        data_format="hf",
        dataset_columns_to_keep=keep,
        num_proc=None,
    )
    assert keep == ["indexed_retention_time"]


def test_cache_cleanup_keeps_the_callers_cache_files(
    download_path_for_assets, tmp_path
):
    # auto_cleanup_cache used Dataset.cleanup_cache_files(), which also deleted the
    # cache files of the caller's own (filtered) source dataset: a second
    # multi-process build from that source then crashed ("One of the subprocesses
    # has abruptly died"), because its workers reopen the deleted files by path
    import os

    source = _private_cache_source(download_path_for_assets, tmp_path)
    source = source.filter(lambda b: [True] * len(b["sequence"]), batched=True)
    source_files = [f["filename"] for f in source.cache_files]
    kwargs = dict(
        data_source=source,
        data_format="hf",
        sequence_column="sequence",
        label_column="intensities",
        model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
        val_ratio=0.2,
        split_seed=1,
        num_proc=2,
    )
    first = FragmentIonIntensityDataset(**kwargs)
    assert all(os.path.exists(f) for f in source_files)
    second = FragmentIonIntensityDataset(**kwargs)  # crashed before the fix
    assert second.extended_alphabet == first.extended_alphabet
    # only the files of the source and of the two final datasets remain: the
    # intermediate files of the processing are still cleaned up
    kept = {f for f in source_files}
    for dataset in (first, second):
        kept |= {
            f["filename"] for s in dataset.hf_dataset.values() for f in s.cache_files
        }
    folder = os.path.dirname(source_files[0])
    cache_files = {
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("cache-") and f.endswith(".arrow")
    }
    assert cache_files <= kept
