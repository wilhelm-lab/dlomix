import os
import sys

import pandas as pd
import tensorflow as tf

from dlomix.data import RetentionTimeDataset
from dlomix.eval import TimeDeltaMetric
from dlomix.models import PrositRetentionTimePredictor

TRAIN_DATAPATH = "example_dataset/proteomTools_train_val.csv"
TEST_DATAPATH = "example_dataset/proteomTools_test.csv"

d = RetentionTimeDataset(
    data_format="csv",
    data_source=TRAIN_DATAPATH,
    test_data_source=TEST_DATAPATH,
    sequence_column="sequence",
    label_column="irt",
    max_seq_len=30,
    batch_size=512,
    val_ratio=0.2,
    dataset_type="pt",
)

print(d)
print(d["train"]["sequence"][0:2])
print(d["train"]["irt"][0:2])

test_targets = d["test"]["irt"]
test_sequences = d["test"]["sequence"]

for x in d.tensor_train_data:
    print(x)
    break


# TODO: Continue with models and training
