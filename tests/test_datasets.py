import logging
import time
from os.path import join
from shutil import rmtree

import pytest
from datasets import Dataset, DatasetDict, load_dataset

from dlomix.data import (
    FragmentIonIntensityDataset,
    RetentionTimeDataset,
    load_processed_dataset,
)

logger = logging.getLogger(__name__)

RT_HUB_DATASET_NAME = "Wilhelmlab/prospect-ptms-irt"


def test_empty_rtdataset():
    rtdataset = RetentionTimeDataset()
    assert rtdataset.hf_dataset is None
    assert rtdataset._empty_dataset_mode is True


def test_parquet_rtdataset():
    rtdataset = RetentionTimeDataset(
        data_source=join(
            pytest.global_variables["DOWNLOAD_PATH_FOR_ASSETS"], "file_1.parquet"
        ),
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


def test_rtdataset_inmemory():
    hf_dataset = load_dataset(
        "parquet",
        data_files=join(
            pytest.global_variables["DOWNLOAD_PATH_FOR_ASSETS"], "file_1.parquet"
        ),
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


def test_csv_rtdataset():
    rtdataset = RetentionTimeDataset(
        data_source=join(
            pytest.global_variables["DOWNLOAD_PATH_FOR_ASSETS"], "file_2.csv"
        ),
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


def test_parquet_intensitydataset():
    filepath = join(
        pytest.global_variables["DOWNLOAD_PATH_FOR_ASSETS"], "file_3.parquet"
    )
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


def test_csv_intensitydataset():
    filepath = join(pytest.global_variables["DOWNLOAD_PATH_FOR_ASSETS"], "file_4.csv")
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


def test_nested_model_features():
    hfdata = Dataset.from_dict(pytest.global_variables["RAW_GENERIC_NESTED_DATA"])

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


def test_save_dataset():
    hfdata = Dataset.from_dict(pytest.global_variables["RAW_GENERIC_NESTED_DATA"])

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
    )

    save_path = "./test_dataset"
    intensity_dataset.save_to_disk(save_path)
    rmtree(save_path)


def test_load_dataset():
    rtdataset = RetentionTimeDataset(
        data_source=join(
            pytest.global_variables["DOWNLOAD_PATH_FOR_ASSETS"], "file_2.csv"
        ),
        data_format="csv",
        sequence_column="sequence",
        label_column="irt",
        val_ratio=0.2,
    )

    save_path = "./test_dataset"
    rtdataset.save_to_disk(save_path)
    splits = rtdataset._data_files_available_splits
    config = rtdataset._config

    load_time_threshold = 0.05  # 50ms

    start_time = time.time()
    loaded_dataset = load_processed_dataset(save_path)
    load_duration = time.time() - start_time
    logger.info("Loaded the dataset in {} seconds".format(load_duration))

    # Assert the load time is below the threshold
    assert (
        load_duration < load_time_threshold
    ), f"Load time exceeded: {load_duration:.3f}s"
    assert loaded_dataset.processed is True

    assert loaded_dataset._data_files_available_splits == splits
    assert loaded_dataset.hf_dataset is not None
    assert loaded_dataset._config == config, f"{loaded_dataset._config} != {config}"
    rmtree(save_path)


def test_no_split_datasetDict_hf_inmemory():
    hfdata = Dataset.from_dict(pytest.global_variables["RAW_GENERIC_NESTED_DATA"])
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
