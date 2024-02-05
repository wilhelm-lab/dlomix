import numpy as np
import pandas as pd

from dlomix.data import IntensityDataset, RetentionTimeDataset

INTENSITY_CSV_EXAMPLE_URL = "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.csv"
TEST_DATA_PARQUET = "metadata.parquet"


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


def test_json_dict_rtdataset():
    test_data_dict = {
        "metadata": {
            "linear rt": [1, 2, 3],
            "modified_sequence": ["ABC", "ABC", "ABC"],
        },
        "annotations": {},
        "parameters": {"target_column_key": "linear rt"},
    }

    pd.DataFrame(test_data_dict["metadata"]).to_parquet(TEST_DATA_PARQUET)

    test_data_dict_file = {
        "metadata": TEST_DATA_PARQUET,
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
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                    ]
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
    intensity_dataset = IntensityDataset(data_source=INTENSITY_CSV_EXAMPLE_URL)

    assert intensity_dataset.sequences is not None
    assert intensity_dataset.collision_energy is not None
    assert intensity_dataset.precursor_charge is not None
    assert intensity_dataset.intensities is not None
    assert intensity_dataset.main_split is IntensityDataset.SPLIT_NAMES[0]
