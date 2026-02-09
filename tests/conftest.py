import glob
import logging
import urllib.request
import zipfile
from os import makedirs
from os.path import exists, join

import pandas as pd
import pytest

logger = logging.getLogger(__name__)

RT_PARQUET_EXAMPLE_URL = "https://zenodo.org/record/6602020/files/TUM_missing_first_meta_data.parquet?download=1"
RT_CSV_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_train_val.csv"
INTENSITY_PARQUET_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.parquet"
INTENSITY_CSV_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.csv"

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


@pytest.fixture(scope="session")
def raw_generic_nested_data():
    return {
        "seq": [
            "[UNIMOD:737]-DASAQTTSHELTIPN-[]",
            "[UNIMOD:737]-DLHTGRLC[UNIMOD:4]-[]",
        ],
        "nested_feature": [[[30, 64]], [[25, 35]]],
        "label": [0.1, 0.2],
        "label2": [1.0, 2.0],
    }


@pytest.fixture(scope="session")
def download_path_for_assets():
    return DOWNLOAD_PATH_FOR_ASSETS


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


@pytest.fixture(scope="session", autouse=True)
def subset_downloaded_files(download_assets):
    # take a subset of the data in each downloaded file
    # to speed up the tests
    N = 100

    files = glob.glob(join(DOWNLOAD_PATH_FOR_ASSETS, "*"))
    logger.info("Subsetting {} files".format(files))
    for i, file in enumerate(files):
        logger.info("Subsetting file {} picking {} samples".format(file, N))
        if ".parquet" in file:
            pd.read_parquet(file).sample(N).to_parquet(file)
        elif ".csv" in file:
            pd.read_csv(file).sample(N).to_csv(file)
        else:
            continue


@pytest.fixture
def basic_alphabet():
    """Basic amino acid alphabet without modifications."""
    return {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "[]-": 21,
        "-[]": 22,
    }


@pytest.fixture
def ptm_alphabet():
    """Alphabet with PTM modifications included."""
    base = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "[]-": 21,
        "-[]": 22,
        "C[UNIMOD:4]": 23,
        "K[UNIMOD:737]": 24,
        "[UNIMOD:737]-": 25,
    }
    return base


@pytest.fixture
def sample_parsed_sequence():
    """Sample parsed sequence data (output of SequenceParsingProcessor)."""
    return {
        "sequence": ["[]-", "D", "E", "L", "-[]"],
        "_parsed_sequence": ["D", "E", "L"],
        "_n_term_mods": "[]-",
        "_c_term_mods": "-[]",
    }


@pytest.fixture
def sample_parsed_sequence_with_ptm():
    """Sample parsed sequence with PTM modifications."""
    return {
        "sequence": ["[]-", "H", "C[UNIMOD:4]", "V", "D", "-[]"],
        "_parsed_sequence": ["H", "C[UNIMOD:4]", "V", "D"],
        "_n_term_mods": "[]-",
        "_c_term_mods": "-[]",
    }


@pytest.fixture
def sample_parsed_sequence_with_nterm_mod():
    """Sample parsed sequence with N-terminal modification."""
    return {
        "sequence": ["[UNIMOD:737]-", "I", "L", "C[UNIMOD:4]", "S", "-[]"],
        "_parsed_sequence": ["I", "L", "C[UNIMOD:4]", "S"],
        "_n_term_mods": "[UNIMOD:737]-",
        "_c_term_mods": "-[]",
    }


@pytest.fixture
def lookup_table():
    # Create a lookup table
    return {
        0: [1.0, 2.0],
        1: [3.0, 4.0],
        2: [5.0, 6.0],
        3: [7.0, 8.0],
    }


@pytest.fixture
def sample_batched_sequences():
    """Sample batch of sequences for batched processor tests."""
    return [
        "[]-DEL-[]",
        "[]-HHDELIF-[]",
        "[]-C[UNIMOD:4]VD-[]",
    ]


@pytest.fixture
def sample_custom_function():
    """Custom function for FunctionProcessor testing."""

    def double_sequence_length(data, **kwargs):
        if isinstance(data.get("sequence"), list):
            return {"sequence": data["sequence"] * 2}
        return data

    return double_sequence_length


@pytest.fixture
def sample_custom_function_with_kwargs():
    """Custom function that accepts kwargs."""

    def scale_feature(data, scale_factor=1.0, **kwargs):
        feature = data.get("feature", 0.0)
        return {"feature": feature * scale_factor}

    return scale_feature
