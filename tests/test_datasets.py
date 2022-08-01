from dlomix.data import RetentionTimeDataset, IntensityDataset
import numpy as np


INTENSITY_CSV_EXAMPLE_URL = 'https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/intensity/intensity_data.csv'
INTENSITY_PARQUET_EXAMPLE_URL = 'https://zenodo.org/record/6602020/files/TUM_missing_first_meta_data.parquet?download=1'

def test_empty_rtdataset():
    rtdataset = RetentionTimeDataset()
    assert rtdataset.sequences is None
    assert rtdataset.targets is None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]

def test_simple_rtdataset():
    rtdataset = RetentionTimeDataset(data_source=(np.array(['AAA', 'BBB']), np.array([21.5, 26.5])))
    assert rtdataset.sequences is not None
    assert rtdataset.targets is not None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]

def test_parquet_rtdataset():
    rtdataset = RetentionTimeDataset(data_source=INTENSITY_PARQUET_EXAMPLE_URL,
                sequence_col='modified_sequence', target_col='indexed_retention_time')
    assert rtdataset.sequences is not None
    assert rtdataset.targets is not None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]    

def test_empty_intensitydataset():
    intensity_dataset = IntensityDataset()
    assert intensity_dataset.sequences is None
    assert intensity_dataset.collision_energy is None
    assert intensity_dataset.precursor_charge is None
    assert intensity_dataset.intensities is None
    assert intensity_dataset.main_split is IntensityDataset.SPLIT_NAMES[0]

def test_simple_intensitydataset():
    intensity_dataset = IntensityDataset(data_source=(
        np.array(['SVFLTFLR']),
        np.array([0.25]),
        np.array([[0, 1, 0, 0, 0, 0]]),
        np.array([[0.03713018032121684, 0.0, -1.0, 0.0, 0.0, -1.0, 0.02485036943573326, 0.0, -1.0, 0.37425569938350733, 0.0, -1.0, 0.1006487907071137, 0.0, -1.0, 0.16793299234113923, 0.0, -1.0, 0.5770605328948204, 0.004866043683849902, -1.0, 0.013969858753800551, 0.0, -1.0, 0.3613063752966507, 0.004158167348899733, -1.0, 0.004756058682204546, 0.0, -1.0, 1.0, 0.05804204277785504, -1.0, 0.0, 0.0, -1.0, 0.0026942297891076857, 0.0042070812435137245, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    ))

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

