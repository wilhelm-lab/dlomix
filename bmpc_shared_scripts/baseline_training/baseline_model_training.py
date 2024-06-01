import argparse
import yaml
import uuid

parser = argparse.ArgumentParser(prog='Baseline Model Training')
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

import os
os.environ['HF_HOME'] = config['dataset']['hf_home']
os.environ['HF_DATASETS_CACHE'] = config['dataset']['hf_cache']

# initialize weights and biases
import wandb
# from wandb.keras import WandbCallback
from wandb.integration.keras import WandbCallback

config['run_id'] = uuid.uuid4()

project_name = f'baseline model training'
wandb.init(
    project=project_name,
    config=config,
    tags=[config['dataset']['name']]
)

# load dataset
from dlomix.data import FragmentIonIntensityDataset

# from misc import PTMS_ALPHABET
from dlomix.constants import PTMS_ALPHABET

from dlomix.data import load_processed_dataset
dataset = load_processed_dataset(wandb.config['dataset']['processed_path'])


# initialize relevant stuff for training
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config['training']['learning_rate'])

from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)


# initialize model
from dlomix.models import PrositIntensityPredictor

input_mapping = {
        "SEQUENCE_KEY": "modified_sequence",
        "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
        "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
        "FRAGMENTATION_TYPE_KEY": "method_nbr",
    }

meta_data_keys=["collision_energy_aligned_normed", "precursor_charge_onehot", "method_nbr"]

model = PrositIntensityPredictor(
    seq_length=wandb.config['dataset']['seq_length'],
    alphabet=PTMS_ALPHABET,
    use_prosit_ptm_features=False,
    with_termini=False,
    input_keys=input_mapping,
    meta_data_keys=meta_data_keys
)

model.compile(
    optimizer=optimizer,
    loss=masked_spectral_distance,
    metrics=[masked_pearson_correlation_distance]
)


# train model
model.fit(
    dataset.tensor_train_data,
    validation_data=dataset.tensor_val_data,
    epochs=wandb.config['training']['num_epochs'],
    callbacks=[WandbCallback(save_model=False, log_batch_frequency=True), early_stopping]
)

out_path = None

if 'save_dir' in wandb.config['model']:
    out_path = f"{wandb.config['model']['save_dir']}/{wandb.config['dataset']['name']}/{wandb.config['run_id']}"

if 'save_path' in wandb.config['model']:
    out_path = wandb.config['model']['save_path']

if out_path is not None:
    model.save(out_path)


# finish up training process
wandb.finish()