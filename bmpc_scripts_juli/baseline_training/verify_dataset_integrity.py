import argparse
import yaml

parser = argparse.ArgumentParser(prog='Baseline Model Training')
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


# load dataset
from dlomix.data import FragmentIonIntensityDataset

# from misc import PTMS_ALPHABET
from dlomix.constants import PTMS_ALPHABET

from dlomix.data import load_processed_dataset
dataset = load_processed_dataset(config['dataset']['processed_path'])


import tensorflow as tf
import numpy as np

min_val = np.Infinity
max_val = -np.Infinity
for x in dataset.tensor_val_data.as_numpy_iterator():
    seq = x[0]['modified_sequence']
    min_val = np.min([min_val, np.min(seq)])
    max_val = np.max([max_val, np.max(seq)])

print(f"min: {min_val}, max: {max_val}")