"""
Build a retention time dataset with PyTorch tensors and print one batch.
For training, see run_prosit_RT_torch.py.

Run from the repository root with the PyTorch backend:
DLOMIX_BACKEND=pytorch python run_scripts/run_prosit_retentiontime_torch.py
"""

from dlomix.data import RetentionTimeDataset

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
