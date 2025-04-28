import logging

import pytest
import torch
from datasets import Dataset

from dlomix.data import FragmentIonIntensityDataset

logger = logging.getLogger(__name__)


def test_dataset_torch():
    hfdata = Dataset.from_dict(pytest.global_variables["RAW_GENERIC_NESTED_DATA"])

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
        dataset_type="pt",
        batch_size=2,
        max_seq_len=15,
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
