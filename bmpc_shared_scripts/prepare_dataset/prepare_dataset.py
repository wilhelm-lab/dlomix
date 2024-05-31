# parse config file
import argparse
import yaml

parser = argparse.ArgumentParser(prog='Prepare a parquet-based dataset for use with DLOmix')
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

import os
os.environ['HF_HOME'] = config['paths']['hf_home']
os.environ['HF_DATASETS_CACHE'] = config['paths']['hf_cache']


# load dataset
from dlomix.data import FragmentIonIntensityDataset
from dlomix.constants import PTMS_ALPHABET

datset_base_path = config['paths']['parquet_path']
dataset_train_path = f"{datset_base_path}_train.parquet"
dataset_val_path = f"{datset_base_path}_val.parquet"
dataset_test_path = f"{datset_base_path}_test.parquet"

dataset = FragmentIonIntensityDataset(
    data_source=dataset_train_path,
    val_data_source=dataset_val_path,
    test_data_source=dataset_test_path,
    data_format="parquet", 
    # val_ratio=config['dataloader']['val_ratio'], # why do we need this if we already have splits?
    batch_size=config['dataloader']['batch_size'],
    max_seq_len=config['dataloader']['seq_length'],
    encoding_scheme="naive-mods",
    alphabet=PTMS_ALPHABET,
    num_proc=config['misc']['num_proc'],
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"]
)


# save processed dataset to disk
dataset.save_to_disk(config['paths']['output_path'])