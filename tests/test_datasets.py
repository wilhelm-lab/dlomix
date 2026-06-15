import logging
import time
from os.path import join
from shutil import rmtree

from datasets import Dataset, DatasetDict, load_dataset

from dlomix.data import (
    FragmentIonIntensityDataset,
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


def test_csv_intensitydataset(download_path_for_assets):
    filepath = join(download_path_for_assets, "file_4.csv")
    intensity_dataset = FragmentIonIntensityDataset(
        data_format="csv",
        data_source=filepath,
        sequence_column="sequence",
        label_column="intensities",
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
    )

    assert intensity_dataset.hf_dataset is not None
    assert intensity_dataset._empty_dataset_mode is False

    example = iter(intensity_dataset.tensor_train_data).next()
    assert example[0]["nested_feature"].shape == [2, 1, 2]


def test_save_dataset(raw_generic_nested_data):
    hfdata = Dataset.from_dict(raw_generic_nested_data)

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
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
    )

    # Get the TensorFlow dataset
    tf_dataset = dataset.tensor_train_data

    # Verify that the TensorFlow dataset is created successfully
    assert tf_dataset is not None
    for batch in tf_dataset.take(1):
        features, labels = batch
        assert features is not None
        assert labels is not None
