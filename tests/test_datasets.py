import logging
import urllib.request
import zipfile
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd
import pytest

from dlomix.data import IntensityDataset, RetentionTimeDataset

logger = logging.getLogger(__name__)

RT_PARQUET_EXAMPLE_URL = "https://zenodo.org/record/6602020/files/TUM_missing_first_meta_data.parquet?download=1"
INTENSITY_CSV_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.csv"

TEST_ASSETS_TO_DOWNLOAD = [
    RT_PARQUET_EXAMPLE_URL,
    INTENSITY_CSV_EXAMPLE_URL,
]
DOWNLOAD_PATH_FOR_ASSETS = join("tests", "assets")


def unzip_file(zip_file_path, dest_dir):
    with zipfile.ZipFile(zip_file_path, "r") as f:
        f.extractall(dest_dir)


@pytest.fixture(scope="session", autouse=True)
def download_assets():
    makedirs(DOWNLOAD_PATH_FOR_ASSETS, exist_ok=True)
    for i, http_address in enumerate(TEST_ASSETS_TO_DOWNLOAD, start=1):
        filename = f"file_{i}"
        filepath = join(DOWNLOAD_PATH_FOR_ASSETS, filename)
        if ".parquet" in http_address:
            filepath += ".parquet"
        if ".csv" in http_address:
            filepath += ".csv"
        if ".zip" in http_address:
            filepath += ".zip"
        if exists(filepath):
            logger.info(
                "Skipping {} since it exists at {}".format(http_address, filepath)
            )
            continue
        logger.info("Downloading: {}, to: {}".format(http_address, filepath))
        urllib.request.urlretrieve(http_address, filepath)
        if ".zip" in filepath:
            logger.info(
                "Unzipping: {}, to: {}".format(filepath, DOWNLOAD_PATH_FOR_ASSETS)
            )
            unzip_file(filepath, DOWNLOAD_PATH_FOR_ASSETS)
    return True


def test_empty_rtdataset():
    rtdataset = RetentionTimeDataset()
    assert rtdataset.sequences is None
    assert rtdataset.targets is None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]


def test_simple_rtdataset():
    rtdataset = RetentionTimeDataset(
        data_source=(np.array(["AAA", "BBB"]), np.array([21.5, 26.5]))
    )
    assert rtdataset.sequences is not None
    assert rtdataset.targets is not None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]


def test_parquet_rtdataset():
    rtdataset = RetentionTimeDataset(
        data_source=join(DOWNLOAD_PATH_FOR_ASSETS, "file_1.parquet"),
        sequence_col="modified_sequence",
        target_col="indexed_retention_time",
    )
    assert rtdataset.sequences is not None
    assert rtdataset.targets is not None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]


def test_json_dict_rtdataset():
    test_data_dict = {
        "metadata": {
            "linear rt": [1, 2, 3],
            "modified_sequence": ["ABC", "ABC", "ABC"],
        },
        "annotations": {},
        "parameters": {"target_column_key": "linear rt"},
    }

    save_path_metadata_parquet = join(DOWNLOAD_PATH_FOR_ASSETS, "metadata.parquet")
    pd.DataFrame(test_data_dict["metadata"]).to_parquet(save_path_metadata_parquet)

    test_data_dict_file = {
        "metadata": save_path_metadata_parquet,
        "annotations": {},
        "parameters": {"target_column_key": "linear rt"},
    }

    rtdataset = RetentionTimeDataset(data_source=test_data_dict, seq_length=20)
    rtdataset_filebased = RetentionTimeDataset(
        data_source=test_data_dict_file, seq_length=20
    )


def test_empty_intensitydataset():
    intensity_dataset = IntensityDataset()
    assert intensity_dataset.sequences is None
    assert intensity_dataset.collision_energy is None
    assert intensity_dataset.precursor_charge is None
    assert intensity_dataset.intensities is None
    assert intensity_dataset.main_split is IntensityDataset.SPLIT_NAMES[0]


def test_simple_intensitydataset():
    intensity_dataset = IntensityDataset(
        data_source=(
            np.array(["SVFLTFLR"]),
            np.array([0.25]),
            np.array([[0, 1, 0, 0, 0, 0]]),
            np.array(
                [
                    [
                        0.03713018032121684,
                        0.0,
                        -1.0,
                        0.0,
                        0.0,
                        -1.0,
                        0.02485036943573326,
                        0.0,
                        -1.0,
                        0.37425569938350733,
                        0.0,
                        -1.0,
                        0.1006487907071137,
                        0.0,
                        -1.0,
                        0.16793299234113923,
                        0.0,
                        -1.0,
                        0.5770605328948204,
                        0.004866043683849902,
                        -1.0,
                        0.013969858753800551,
                        0.0,
                        -1.0,
                        0.3613063752966507,
                        0.004158167348899733,
                        -1.0,
                        0.004756058682204546,
                        0.0,
                        -1.0,
                        1.0,
                        0.05804204277785504,
                        -1.0,
                        0.0,
                        0.0,
                        -1.0,
                        0.0026942297891076857,
                        0.0042070812435137245,
                        -1.0,
                        0.0,
                        0.0,
                    ]
                    + [-1.0] * 132
                ]
            ),
        )
    )

    assert intensity_dataset.sequences is not None
    assert intensity_dataset.collision_energy is not None
    assert intensity_dataset.precursor_charge is not None
    assert intensity_dataset.intensities is not None
    assert intensity_dataset.main_split is IntensityDataset.SPLIT_NAMES[0]


def test_csv_intensitydataset():
    filepath = join(DOWNLOAD_PATH_FOR_ASSETS, "file_2.csv")
    intensity_dataset = IntensityDataset(data_source=filepath, sample_run=True)

    assert intensity_dataset.sequences is not None
    assert intensity_dataset.collision_energy is not None
    assert intensity_dataset.precursor_charge is not None
    assert intensity_dataset.intensities is not None
    assert intensity_dataset.main_split is IntensityDataset.SPLIT_NAMES[0]
