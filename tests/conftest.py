import logging
import urllib.request
import zipfile
from os import makedirs
from os.path import exists, join

import pytest

logger = logging.getLogger(__name__)

RT_PARQUET_EXAMPLE_URL = "https://zenodo.org/record/6602020/files/TUM_missing_first_meta_data.parquet?download=1"
RT_CSV_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_train_val.csv"
INTENSITY_PARQUET_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.parquet"
INTENSITY_CSV_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.csv"
RT_HUB_DATASET_NAME = "Wilhelmlab/prospect-ptms-irt"


RAW_GENERIC_NESTED_DATA = {
    "seq": ["[UNIMOD:737]-DASAQTTSHELTIPN-[]", "[UNIMOD:737]-DLHTGRLC[UNIMOD:4]-[]"],
    "nested_feature": [[[30, 64]], [[25, 35]]],
    "label": [0.1, 0.2],
}

TEST_ASSETS_TO_DOWNLOAD = [
    RT_PARQUET_EXAMPLE_URL,
    RT_CSV_EXAMPLE_URL,
    INTENSITY_PARQUET_EXAMPLE_URL,
    INTENSITY_CSV_EXAMPLE_URL,
]

DOWNLOAD_PATH_FOR_ASSETS = join("tests", "assets")


def unzip_file(zip_file_path, dest_dir):
    with zipfile.ZipFile(zip_file_path, "r") as f:
        f.extractall(dest_dir)


@pytest.fixture(scope="session", autouse=True)
def global_variables():
    pytest.global_variables = {
        "RAW_GENERIC_NESTED_DATA": RAW_GENERIC_NESTED_DATA,
        "DOWNLOAD_PATH_FOR_ASSETS": DOWNLOAD_PATH_FOR_ASSETS,
    }


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
