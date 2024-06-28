import os
os.environ['HF_HOME'] = "/cmnfs/proj/bmpc_dlomix/datasets"
os.environ['HF_DATASETS_CACHE'] = "/cmnfs/proj/bmpc_dlomix/datasets/hf_cache"

# hyperparameters
config = {}

config['seq_length'] = 30
config['batch_size'] = 1024
config['val_ratio'] = 0.2
config['num_proc'] = 40
# config['learning_rate'] = 1.0e-4
# config['epochs'] = 20

# load dataset
from dlomix.data import FragmentIonIntensityDataset

# from misc import PTMS_ALPHABET
from dlomix.constants import PTMS_ALPHABET

from datasets import disable_caching
# disable_caching()

# path to dataset
datset_base_path = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug"
# datset_base_path = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/clean"
dataset_train_path = f"{datset_base_path}_train.parquet"
dataset_val_path = f"{datset_base_path}_val.parquet"
dataset_test_path = f"{datset_base_path}_test.parquet"

dataset = FragmentIonIntensityDataset(
    data_source=dataset_train_path,
    val_data_source=dataset_val_path,
    test_data_source=dataset_test_path,
    data_format="parquet", 
    val_ratio=config['val_ratio'], # why do we need this if we already have splits?
    batch_size=config['batch_size'],
    max_seq_len=config['seq_length'],
    encoding_scheme="naive-mods",
    alphabet=PTMS_ALPHABET,
    num_proc=config['num_proc'],
    # model_features=[]
    model_features=["precursor_charge_onehot", "collision_energy_aligned_normed","method_nbr"]
)


dataset.save_to_disk(f"/cmnfs/proj/bmpc_dlomix/processed/ptm_baseline_bs{config['batch_size']}")