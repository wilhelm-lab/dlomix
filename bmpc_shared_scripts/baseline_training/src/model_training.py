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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler


def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def model_training(config):
    def run():
        config['run_id'] = uuid.uuid4()

        # initialize weights and biases
        wandb.init(
            project=config["project"],
            config=config,
            tags=[config['dataset']['name']]
        )

        if 'cuda_device_nr' in wandb.config['processing']:
            os.environ["CUDA_VISIBLE_DEVICES"] = wandb.config['processing']['cuda_device_nr']

        if 'num_proc' in wandb.config['processing']:
            num_proc = wandb.config['processing']['num_proc']
            os.environ["OMP_NUM_THREADS"] = f"{num_proc}"
            os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_proc}"
            os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_proc}"
            tf.config.threading.set_inter_op_parallelism_threads(
                num_proc
            )
            tf.config.threading.set_intra_op_parallelism_threads(
                num_proc
            )

        os.environ['HF_HOME'] = wandb.config['dataset']['hf_home']
        os.environ['HF_DATASETS_CACHE'] = wandb.config['dataset']['hf_cache']

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

        if 'lr_warmup_linear' in wandb.config['training']:
            print("using lr warmup linear")
            num_epochs = wandb.config['training']['lr_warmup_linear']['num_epochs']
            start_lr = wandb.config['training']['lr_warmup_linear']['start_lr']
            end_lr = wandb.config['training']['lr_warmup_linear']['end_lr']
            def scheduler(epoch, lr):
                if epoch < num_epochs:
                    print("warmup step")
                    factor = epoch / num_epochs
                    return factor * end_lr + (1-factor) * start_lr
                else:
                    return lr
            
            lr_warmup_linear = LearningRateScheduler(scheduler)
            callbacks.append(lr_warmup_linear)


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

    return run



# combines nested dicts
# source: https://stackoverflow.com/questions/70310388/how-to-merge-nested-dictionaries/70310511#70310511
def combine_into(d: dict, combined: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            combine_into(v, combined.setdefault(k, {}))
        else:
            combined[k] = v