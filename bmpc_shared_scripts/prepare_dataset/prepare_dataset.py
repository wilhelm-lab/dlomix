# parse config file
import argparse
import yaml
import os

parser = argparse.ArgumentParser(prog='Prepare a parquet-based dataset for use with DLOmix')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--num-proc', type=int, required=False)
args = parser.parse_args()

with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

if args.num_proc is not None:
    config.setdefault('processing', {})['num_proc'] = args.num_proc

os.environ['HF_HOME'] = config['dataset']['hf_home']
os.environ['HF_DATASETS_CACHE'] = config['dataset']['hf_cache']

# these imports need to come after setting the HF_ environment variables
from dlomix.data import FragmentIonIntensityDataset
from dlomix.constants import PTMS_ALPHABET, ALPHABET_UNMOD, ALPHABET_NAIVE_MODS


# select alphabet
if isinstance(config['dataset']['alphabet'], dict):
    # this is a custom alphabet
    alphabet = config['dataset']['alphabet']
elif config['dataset']['alphabet'] == 'PTMS_ALPHABET':
    alphabet = PTMS_ALPHABET
elif config['dataset']['alphabet'] == 'ALPHABET_UNMOD':
    alphabet = ALPHABET_UNMOD
elif config['dataset']['alphabet'] == 'ALPHABET_NAIVE_MODS':
    alphabet = ALPHABET_NAIVE_MODS
else:
    raise ValueError('unknown alphabet selected')



# load dataset
datset_base_path = config['dataset']['parquet_path']

if datset_base_path.endswith('.parquet'):
    dataset_train_path = datset_base_path
    dataset_val_path, dataset_test_path = None, None
else:
    dataset_train_path = f"{datset_base_path}_train.parquet"
    dataset_val_path = f"{datset_base_path}_val.parquet"
    dataset_test_path = f"{datset_base_path}_test.parquet"

dataset = FragmentIonIntensityDataset(
    data_source=dataset_train_path,
    val_data_source=dataset_val_path,
    test_data_source=dataset_test_path,
    data_format="parquet", 
    val_ratio=config['dataset'].get('val_ratio', 0.2), # why do we need this if we already have splits?
    batch_size=config['dataset']['batch_size'],
    max_seq_len=config['dataset']['seq_length'],
    encoding_scheme="naive-mods",
    alphabet=alphabet,
    num_proc=config['processing']['num_proc'],
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"]
)


# save processed dataset to disk
dataset.save_to_disk(config['dataset']['processed_path'])