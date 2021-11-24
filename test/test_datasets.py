from dlomix.data import RetentionTimeDataset


def test_empty_rtdataset():
    rtdataset = RetentionTimeDataset()
    assert rtdataset.sequences is None
    assert rtdataset.targets is None
    assert rtdataset.main_split is RetentionTimeDataset.SPLIT_NAMES[0]
