from dlomix.data import RetentionTimeDataset
import numpy as np

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
