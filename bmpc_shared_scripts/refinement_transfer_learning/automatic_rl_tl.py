import yaml
import os
import uuid

import wandb
from wandb.integration.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import change_layers
import freezing
from recompile_callbacks import *

from dlomix.constants import PTMS_ALPHABET, ALPHABET_NAIVE_MODS, ALPHABET_UNMOD
from dlomix.data import load_processed_dataset, FragmentIonIntensityDataset
from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

from dataclasses import dataclass, asdict
from typing import Optional

class Dataset:
    is_preprocessed : bool
    parquet_path : Optional[str]
    preprocessed_path : Optional[str]

    def __init__(self, *, preprocessed_path : Optional[str] = None, parquet_path : Optional[str] = None):
        if preprocessed_path is not None and parquet_path is not None:
            raise ValueError("Either specify parquet_path or preprocessed_path")

        if preprocessed_path is not None:
            self.is_preprocessed = True
            self.preprocessed_path = preprocessed_path
        else:
            self.is_preprocessed = False
            self.parquet_path = parquet_path


@dataclass
class AutomaticRlTlTrainingConfig:
    # dataset/model parameters
    dataset : Dataset
    model_path : Optional[str] = None

    # wandb parameters
    use_wandb : bool = False
    wandb_project : str = 'DLOmix_auto_RL_TL'
    wandb_tags : list[str] = []

    # tensorflow parameters
    tf_cuda_devices : Optional[list[int]] = None
    tf_num_procs : Optional[int] = None
    hf_home_directory : Optional[str] = None
    hf_cache_directory : Optional[str] = None


    def to_dict(self):
        """Converts configuration to a python dict object.

        Returns:
            dict: Configuration options as dictionary
        """
        return asdict(self)


class AutomaticRlTlTraining:
    config : AutomaticRlTlTrainingConfig

    dataset : FragmentIonIntensityDataset
    model : PrositIntensityPredictor
    is_new_model : bool = False

    requires_new_embedding_layer : bool = False
    can_reuse_old_embedding_weights : bool = True

    current_epoch_offset : int = 0

    def __init__(self, config : AutomaticRlTlTrainingConfig):
        self.config = config

        self._init_wandb()
        self._init_tensorflow()
        self._load_dataset()
        self._init_model()
    
    def _init_wandb(self):
        """ Initializes Weights & Biases Logging if the user requested that in the config.
        """
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.to_dict(),
                tags=self.config.wandb_tags
            )

    def _init_tensorflow(self):
        """Initializes Tensorflow based on the parameters in the config.
        """
        if self.config.tf_cuda_devices is not None:
            cuda_device_str = '-1'
            if len(self.config.tf_cuda_devices) > 0:
                cuda_device_str = ','.join([str(x) for x in self.config.tf_cuda_devices])

            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_str

        if self.config.tf_num_procs is not None:
            num_proc = self.config.tf_num_procs
            os.environ["OMP_NUM_THREADS"] = f"{num_proc}"
            os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_proc}"
            os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_proc}"
            tf.config.threading.set_inter_op_parallelism_threads(
                num_proc
            )
            tf.config.threading.set_intra_op_parallelism_threads(
                num_proc
            )

        if self.config.hf_home_directory is not None:
            os.environ['HF_HOME'] = self.config.hf_home_directory

        if self.config.hf_cache_directory is not None:
            os.environ['HF_DATASETS_CACHE'] = self.config.hf_cache_directory
        
    def _load_dataset(self):
        """Loads the specified dataset.

        Raises:
            NotImplementedError: Raised if an unprocessed dataset is provided
        """
        # TODO: should we support loading a dataset which is not preprocessed yet here
        if not self.config.dataset.is_preprocessed:
            raise NotImplementedError("Only preprocessed datasets are currently supported by the automatic RL/TL training.")

        self.dataset = load_processed_dataset(self.config.dataset.preprocessed_path)

    def _init_model(self):
        if self.config.model_path is not None:
            print(f"loading model from file {self.config.model_path}")
            self.model = tf.keras.models.load_model(self.config.model_path)

        else:
            # initialize new model
            input_mapping = {
                    "SEQUENCE_KEY": "modified_sequence",
                    "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
                    "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
                    "FRAGMENTATION_TYPE_KEY": "method_nbr",
                }

            meta_data_keys=["collision_energy_aligned_normed", "precursor_charge_onehot", "method_nbr"]

            self.model = PrositIntensityPredictor(
                seq_length=self.dataset.max_seq_len,
                alphabet=self.dataset.alphabet,
                use_prosit_ptm_features=False,
                with_termini=False,
                input_keys=input_mapping,
                meta_data_keys=meta_data_keys
            )

    def _update_model_inputs(self):
        model_alphabet = self.model.alphabet
        dataset_alphabet = self.dataset.alphabet

        if self.is_new_model:
            print('[embedding layer]  created new model with fresh embedding layer')
            return

        if model_alphabet == dataset_alphabet:
            print('[embedding layer]  model and dataset modifications match')
        else:
            print('[embedding layer]  model and dataset modifications do not match')
            self.requires_new_embedding_layer = True
            # check if the existing embedding can be reused
            including_entries = [model_val == dataset_alphabet[key] for key, model_val in model_alphabet.items()]
            if all(including_entries):
                print('[embedding layer]  can reuse old embedding weights')
            else:
                print('[embedding layer]  old embedding weights cannot be reused (mismatch in the mapping)')
                self.can_reuse_old_embedding_weights = False

    def _update_model_outputs(self):
        # check that sequence length matches
        if self.model.seq_len != self.dataset.max_seq_len:
            raise RuntimeError(f"Max. sequence length does not match between dataset and model (dataset: {self.dataset.max_seq_len}, model: {self.model.seq_len})")

        # check whether number of ions matches
        model_ions = self.model.len_fion
        dataset_ions = self.dataset

        # TODO: check if number of ions matches


class AutomaticRlTlTrainingInstance:

    config : AutomaticRlTlTrainingConfig
    current_epoch_offset : int


    def __init__(self, config : AutomaticRlTlTrainingConfig, current_epoch_offset : int):
        self.config = config
        self.current_epoch_offset = current_epoch_offset

    """
    def configure_training(self):
        # initialize relevant stuff for training
        self.total_epochs = wandb.config['training']['num_epochs']
        self.recompile_callbacks = [RecompileCallback(
            epoch=self.total_epochs,
            callback=lambda *args: None
        )]

        # refinement/transfer learning configuration
        rl_config = wandb.config['refinement_transfer_learning']
        if rl_config is None:
            rl_config = {}

        # optionally: replacing of input/output layers
        if 'new_output_layer' in rl_config:
            change_layers.change_output_layer(self.model, rl_config['new_output_layer']['num_ions'])
        if 'new_input_layer' in rl_config:
            new_alphabet = self.get_alphabet(rl_config['new_input_layer']['new_alphabet'])
            change_layers.change_input_layer(
                self.model,
                new_alphabet,
                rl_config['new_input_layer']['freeze_old_weights']
            )

            if rl_config['new_input_layer']['freeze_old_weights']:
                wandb.log({'freeze_old_embedding_weights': 1})

                def release_callback():
                    change_layers.release_old_embeddings(self.model)
                    wandb.log({'freeze_old_embedding_weights': 0})

                self.recompile_callbacks.append(RecompileCallback(
                    epoch=rl_config['new_input_layer']['release_after_epochs'],
                    callback=release_callback
                ))

        
        # optionally: freeze layers during training
        if 'freeze_layers' in rl_config:
            if 'activate' not in rl_config['freeze_layers'] or rl_config['freeze_layers']['activate']:
                print('freezing active')
                freezing.freeze_model(
                    self.model, 
                    rl_config['freeze_layers']['is_first_layer_trainable'],
                    rl_config['freeze_layers']['is_last_layer_trainable']
                )
                wandb.log({'freeze_layers': 1})

                def release_callback():
                    freezing.release_model(self.model)
                    wandb.log({'freeze_layers': 0})

                self.recompile_callbacks.append(RecompileCallback(
                    epoch=rl_config['freeze_layers']['release_after_epochs'],
                    callback=release_callback
                ))


        class LearningRateReporter(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, *args):
                wandb.log({'learning_rate': self.model.optimizer._learning_rate.numpy()})

        class RealEpochReporter(tf.keras.callbacks.Callback):
            def on_epoch_begin(self_inner, epoch, *args):
                wandb.log({'epoch_total': epoch + self.current_epoch_offset})

        self.callbacks = [WandbCallback(save_model=False, log_batch_frequency=True, verbose=1), LearningRateReporter(), RealEpochReporter()]

        if 'early_stopping' in wandb.config['training']:
            print("using early stopping")
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=wandb.config['training']['early_stopping']['min_delta'],
                patience=wandb.config['training']['early_stopping']['patience'],
                restore_best_weights=True)

            self.callbacks.append(early_stopping)

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

            self.callbacks.append(reduce_lr)

        if 'lr_warmup_linear' in wandb.config['training']:
            print("using lr warmup linear")
            num_epochs = wandb.config['training']['lr_warmup_linear']['num_epochs']
            start_lr = wandb.config['training']['lr_warmup_linear']['start_lr']
            end_lr = wandb.config['training']['lr_warmup_linear']['end_lr']
            def scheduler(epoch, lr):
                global_epoch = epoch + self.current_epoch_offset
                if global_epoch < num_epochs:
                    print("warmup step")
                    factor = global_epoch / num_epochs
                    return factor * end_lr + (1-factor) * start_lr
                else:
                    return lr
            
            lr_warmup_linear = LearningRateScheduler(scheduler)
            self.callbacks.append(lr_warmup_linear)

    def perform_training(self):
        # perform all training runs
        current_learning_rate = wandb.config['training']['learning_rate']
        for training_part in get_training_parts(self.recompile_callbacks, self.total_epochs):
            # (re-)compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=current_learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss=masked_spectral_distance,
                metrics=[masked_pearson_correlation_distance]
            )

            if training_part.num_epochs > 0:
                # train model
                history = self.model.fit(
                    self.dataset.tensor_train_data,
                    validation_data=self.dataset.tensor_val_data,
                    epochs=training_part.num_epochs,
                    callbacks=self.callbacks
                )

                if len(history.history['loss']) < training_part.num_epochs:
                    # early stopping
                    break

            # call callbacks
            training_part()
            
            current_learning_rate = self.model.optimizer._learning_rate.numpy()
            self.current_epoch_offset += training_part.num_epochs

    def save_model(self):

        out_path = None
        if 'save_dir' in wandb.config['model']:
            out_path = f"{wandb.config['model']['save_dir']}/{wandb.config['dataset']['name']}/{wandb.config['run_id']}.keras"
        if 'save_path' in wandb.config['model']:
            out_path = wandb.config['model']['save_path']

        if out_path is not None:
            dir = os.path.dirname(out_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            
            print(f'saving the model to {out_path}')
            self.model.save(out_path)
            # self.model.save('test.keras')

    def __call__(self):
        # setting up training
        self.init_config()
        self.load_dataset()
        self.initialize_model()
        self.configure_training()

        # do training
        self.perform_training()
        
        # finish up
        self.save_model()
        wandb.finish()

    """
