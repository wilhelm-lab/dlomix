import argparse
import yaml
import os
import uuid

import wandb
from wandb.integration.keras import WandbCallback

from dlomix.constants import PTMS_ALPHABET
from dlomix.data import load_processed_dataset
from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# parse args
parser = argparse.ArgumentParser(prog='Baseline Model Training')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--sweep-id', type=str, required=True)
parser.add_argument('--tf-device-nr', type=str, required=True)
parser.add_argument('--count', type=int, required=False)
args = parser.parse_args()

with open(args.config, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

os.environ['HF_HOME'] = config['dataset']['hf_home']
os.environ['HF_DATASETS_CACHE'] = config['dataset']['hf_cache']

os.environ["CUDA_VISIBLE_DEVICES"] = args.tf_device_nr

def run():
    config['run_id'] = uuid.uuid4()

    # initialize weights and biases
    project_name = f'baseline model training'
    wandb.init(
        project=project_name,
        config=config,
        tags=[config['dataset']['name']]
    )


    # load dataset
    dataset = load_processed_dataset(wandb.config['dataset']['processed_path'])


    # initialize relevant stuff for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config['training']['learning_rate'])

    # load or create model
    if 'load_path' in wandb.config['model']:
        print(f"loading model from file {wandb.config['model']['load_path']}")
        model = tf.keras.models.load_model(wandb.config['model']['load_path'])
    else:
        # initialize model
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

    class LearningRateReporter(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, *args):
            wandb.log({'learning_rate': self.model.optimizer._learning_rate.numpy()})

    callbacks = [WandbCallback(save_model=False, log_batch_frequency=True, verbose=1), LearningRateReporter()]

    if 'early_stopping' in wandb.config['training']:
        print("using early stopping")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=wandb.config['training']['early_stopping']['min_delta'],
            patience=wandb.config['training']['early_stopping']['patience'],
            restore_best_weights=True)

        callbacks.append(early_stopping)

    if 'lr_scheduler_plateau' in wandb.config['training']:
        print("using lr scheduler plateau")
        # Reduce LR on Plateau Callback
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=wandb.config['training']['lr_scheduler_plateau']['factor'],
            patience=wandb.config['training']['lr_scheduler_plateau']['patience'],
            min_delta=wandb.config['training']['lr_scheduler_plateau']['min_delta'],
            cooldown=wandb.config['training']['lr_scheduler_plateau']['cooldown']
        ) 

        callbacks.append(reduce_lr)

    # train model
    model.fit(
        dataset.tensor_train_data,
        validation_data=dataset.tensor_val_data,
        epochs=wandb.config['training']['num_epochs'],
        callbacks=callbacks
    )

    out_path = None

    if 'save_dir' in wandb.config['model']:
        out_path = f"{wandb.config['model']['save_dir']}/{wandb.config['dataset']['name']}/{wandb.config['run_id']}.keras"

    if 'save_path' in wandb.config['model']:
        out_path = wandb.config['model']['save_path']

    if out_path is not None:
        model.save(out_path)


    # finish up training process
    wandb.finish()


# start agent
wandb.agent(args.sweep_id, run, count=args.count)