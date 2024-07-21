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
    input_model_path : Optional[str] = None
    output_model_path : str

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

@dataclass
class TrainingInstanceConfig:
    learning_rate : float
    num_epochs : int
    
    freeze_inner_layers : bool
    freeze_whole_embedding_layer : bool
    freeze_whole_regressor_layer : bool
    freeze_old_embedding_weights : bool
    freeze_old_regressor_weights : bool

    plateau_early_stopping : bool
    plateau_early_stopping_patience : int
    plateau_early_stopping_min_delta : float

    inflection_early_stopping : bool
    # TODO 
    inflection_early_stopping_

    lr_scheduler_plateau : bool
    lr_scheduler_plateau_factor : float
    lr_scheduler_plateau_min_delta : float
    lr_scheduler_plateau_patience : int
    lr_scheduler_plateau_cooldown : int

    lr_warmup : bool
    # lr_warmup_type : str = 'linear'
    lr_warmup_num_epochs : int
    lr_warmup_start_lr : float


class AutomaticRlTlTraining:
    config : AutomaticRlTlTrainingConfig

    dataset : FragmentIonIntensityDataset
    model : PrositIntensityPredictor
    is_new_model : bool = False

    requires_new_embedding_layer : bool
    can_reuse_old_embedding_weights : bool
    requires_new_regressor_layer : bool
    can_reuse_old_regressor_weights : bool
    
    current_epoch_offset : int = 0
    callbacks : list = []

    def __init__(self, config : AutomaticRlTlTrainingConfig):
        self.config = config

        self._init_wandb()
        self._init_tensorflow()
        self._load_dataset()
        self._init_model()
        self._update_model_inputs()
        self._update_model_outputs()
        self._init_training()
    
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
            self.requires_new_embedding_layer = False
        else:
            print('[embedding layer]  model and dataset modifications do not match')
            self.requires_new_embedding_layer = True
            # check if the existing embedding can be reused
            including_entries = [model_val == dataset_alphabet[key] for key, model_val in model_alphabet.items()]
            if all(including_entries):
                print('[embedding layer]  can reuse old embedding weights')
                self.can_reuse_old_embedding_weights = True
            else:
                print('[embedding layer]  old embedding weights cannot be reused (mismatch in the mapping)')
                self.can_reuse_old_embedding_weights = False

            change_layers.change_input_layer(
                self.model,
                dataset_alphabet,
                freeze_old_embeds=self.can_reuse_old_embedding_weights
            )
            self.model.alphabet = dataset_alphabet

    def _update_model_outputs(self):
        # check that sequence length matches
        if self.model.seq_len != self.dataset.max_seq_len:
            raise RuntimeError(f"Max. sequence length does not match between dataset and model (dataset: {self.dataset.max_seq_len}, model: {self.model.seq_len})")

        # check whether number of ions matches
        model_ions = ['y', 'b']
        if hasattr(self.model, 'ion_types') and self.model.ion_types is not None:
            model_ions = self.model.ion_types 

        dataset_ions = ['y', 'b']
        if hasattr(self.dataset, 'ion_types') and self.dataset.ion_types is not None:
            dataset_ions = self.dataset.ion_types 

        if model_ions == dataset_ions:
            print('[regressor layer]  matching ion types')
            self.requires_new_regressor_layer = True
        else:
            print('[regressor layer]  ion types not matching')
            self.requires_new_regressor_layer = False

            if len(model_ions) <= len(dataset_ions) and all([m == d for m, d in zip(model_ions, dataset_ions)]):
                print('[regressor layer]  can reuse existing regressor weights')
                self.can_reuse_old_regressor_weights = True
            else:
                print('[regressor layer]  old regressor weights cannot be reused (mismatch in the ion ordering / num. ions)')
                self.can_reuse_old_regressor_weights = False
            
            change_layers.change_output_layer(
                self.model,
                len(dataset_ions),
                freeze_old_output=self.can_reuse_old_regressor_weights
            )
            self.model.ion_types = dataset_ions

   
    def _init_training(self):
        if self.config.use_wandb:
            class LearningRateReporter(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, *args):
                    wandb.log({'learning_rate': self.model.optimizer._learning_rate.numpy()})

            class RealEpochReporter(tf.keras.callbacks.Callback):
                def on_epoch_begin(self_inner, epoch, *args):
                    wandb.log({'epoch_total': epoch + self.current_epoch_offset})

            self.callbacks = [WandbCallback(save_model=False, log_batch_frequency=True, verbose=1), LearningRateReporter(), RealEpochReporter()]


    def run(self):

        training_hierachy = [

        ]

        for instance_config in training_hierachy:
            training = AutomaticRlTlTrainingInstance(instance_config, self.current_epoch_offset, self.config.use_wandb, self.callbacks)
            training.run()

            self.current_epoch_offset = training.current_epoch_offset

        self._save_model()

        if self.config.use_wandb:
            wandb.finish()


    def _save_model(self):
        print(f'saving the model to {self.config.output_model_path}')
        self.model.save(self.config.output_model_path)



class AutomaticRlTlTrainingInstance:

    instance_config : TrainingInstanceConfig
    current_epoch_offset : int
    wandb_logging : bool
    callbacks : list

    stopped_early : bool
    final_learning_rate : float


    def __init__(self, instance_config : TrainingInstanceConfig, current_epoch_offset : int, wandb_logging : bool, callbacks : list):
        self.instance_config = instance_config
        self.current_epoch_offset = current_epoch_offset
        self.wandb_logging = wandb_logging
        self.callbacks = callbacks.copy()

        self._configure_training()


    def _configure_training(self):

        # freezing of old embedding weights
        if self.instance_config.freeze_old_embedding_weights:
            if self.wandb_logging:
                wandb.log({'freeze_old_embedding_weights': 1})
            change_layers.freeze_old_embeddings(self.model)
        else:
            if self.wandb_logging:
                wandb.log({'freeze_old_embedding_weights': 0})
            change_layers.release_old_embeddings(self.model)
        
        # freezing of old regressor weights
        if self.instance_config.freeze_old_regressor_weights:
            if self.wandb_logging:
                wandb.log({'freeze_old_regressor_weights': 1})
            change_layers.freeze_old_regressor(self.model)
        else:
            if self.wandb_logging:
                wandb.log({'freeze_old_regressor_weights': 0})
            change_layers.release_old_regressor(self.model)


        # freezing of inner layers
        if self.instance_config.freeze_inner_layers:
            freezing.freeze_model(
                self.model, 
                self.instance_config.freeze_whole_embedding_layer,
                self.instance_config.freeze_whole_regressor_layer
            )

            if self.wandb_logging:
                wandb.log({
                    'freeze_layers': 1,
                    'freeze_embedding_layer': 1 if self.instance_config.freeze_whole_embedding_layer else 0,
                    'freeze_regressor_layer': 1 if self.instance_config.freeze_whole_regressor_layer else 0
                })
        else:
            if self.instance_config.freeze_whole_embedding_layer:
                raise RuntimeError('Cannot freeze whole embedding layer without freezing inner part of the model.')
            if self.instance_config.freeze_whole_regressor_layer:
                raise RuntimeError('Cannot freeze whole regressor layer without freezing inner part of the model.')
            
            freezing.release_model(self.model)

            if self.wandb_logging:
                wandb.log({
                    'freeze_layers': 0,
                    'freeze_embedding_layer': 0,
                    'freeze_regressor_layer': 0    
                })


        if self.instance_config.plateau_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=self.instance_config.plateau_early_stopping_min_delta,
                patience=self.instance_config.plateau_early_stopping_patience,
                restore_best_weights=True)

            self.callbacks.append(early_stopping)


        if self.instance_config.inflection_early_stopping:
            class InflectionPointEarlyStopping(tf.keras.callbacks.Callback):
                min_delta : float

                def __init__(self, min_delta : float, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.min_delta = min_delta


                def on_train_batch_end(self, batch, *args):
                    # TODO: implement                    


        if self.instance_config.lr_scheduler_plateau:
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.instance_config.lr_scheduler_plateau_factor,
                patience=self.instance_config.lr_scheduler_plateau_patience,
                min_delta=self.instance_config.lr_scheduler_plateau_min_delta,
                cooldown=self.instance_config.lr_scheduler_plateau_cooldown
            ) 

            self.callbacks.append(reduce_lr)

        if self.instance_config.lr_warmup:
            num_epochs = self.instance_config.lr_warmup_num_epochs
            start_lr = self.instance_config.lr_warmup_start_lr
            end_lr = self.instance_config.learning_rate
            def scheduler(epoch, lr):
                global_epoch = epoch + self.current_epoch_offset
                if global_epoch < num_epochs:
                    factor = global_epoch / num_epochs
                    return factor * end_lr + (1-factor) * start_lr
                else:
                    return lr
            
            lr_warmup_linear = LearningRateScheduler(scheduler)
            self.callbacks.append(lr_warmup_linear)


    def run(self):
        # perform all training runs
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.instance_config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance]
        )

        # train model
        history = self.model.fit(
            self.dataset.tensor_train_data,
            validation_data=self.dataset.tensor_val_data,
            epochs=self.instance_config.num_epochs,
            callbacks=self.callbacks
        )

        if len(history.history['loss']) < self.instance_config.num_epochs:
            self.stopped_early = True
        else:
            self.stopped_early = False

        self.final_learning_rate = self.model.optimizer._learning_rate.numpy()
        self.current_epoch_offset += len(history.history['loss'])

