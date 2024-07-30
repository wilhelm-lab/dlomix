import yaml
import os
import uuid

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from .custom_callbacks import InflectionPointEarlyStopping, LearningRateWarmupPerStep, InflectionPointLRReducer

from dlomix.refinement_transfer_learning import change_layers, freezing
from dlomix.constants import PTMS_ALPHABET, ALPHABET_NAIVE_MODS, ALPHABET_UNMOD
from dlomix.data import load_processed_dataset, FragmentIonIntensityDataset
from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance

from dataclasses import dataclass, asdict, field
from typing import Optional
import math


@dataclass
class AutomaticRlTlTrainingConfig:
    """Configuration for an automatic refinement/transfer learning run.

    Attributes:
        dataset (FragmentIonIntensityDataset): Dataset that should be used for training. The datasets needs a train and validation split and must not be an inference-only dataset.
        baseline_model (Optional[PrositIntensityPredictor]): If a model is provided, this model is used as baseline for training. If no model is specified, a new model is trained from scratch.
        min_warmup_sequences_new_weights (int): Determines, the length the learning rate warmup phase in phase 1 of the automatic training pipeline (training of newly added weights). Default: 4000000
        min_warmup_sequences_whole_model (int): Determines, the length the learning rate warmup phase in phase 2 of the automatic training pipeline (training of all weights in the model). Default: 4000000
        improve_further (bool): Determines whether a third training phase is performed which has more restrictive early stopping criterions and learning rate scheduling. Default: True
        use_wandb (bool): Determines whether to use wandb to log the training run. Wandb needs to be installed as dependency if this is set to True. Default: False
        wandb_project (str): Selects the wandb project that the run should correspond to. This is ignored if use_wandb is set to False. Default: "DLOmix_auto_RL_TL"
        wandb_tags (list[str]): List of wandb tags to add to the run. This is ignored if use_wandb is set to False. Default: [] 
    """

    # dataset/model parameters
    dataset : FragmentIonIntensityDataset 
    baseline_model : Optional[PrositIntensityPredictor]

    # training parameters
    min_warmup_sequences_new_weights : int = 4000000
    min_warmup_sequences_whole_model : int = 4000000
    improve_further : bool = True

    # wandb parameters
    use_wandb : bool = False
    wandb_project : str = 'DLOmix_auto_RL_TL'
    wandb_tags : list[str] = field(default_factory=list)

    
    def to_dict(self):
        """Converts configuration to a python dict object. Only attributes are included which can be easily represented as text.

        Returns:
            dict: Configuration options as dictionary
        """
        return {
            'min_warmup_sequences_new_weights': self.min_warmup_sequences_new_weights,
            'min_warmup_sequences_whole_model': self.min_warmup_sequences_whole_model,
            'improve_further': self.improve_further
        }


@dataclass
class TrainingInstanceConfig:
    learning_rate : float
    num_epochs : int
    
    freeze_inner_layers : bool = False
    freeze_whole_embedding_layer : bool = False
    freeze_whole_regressor_layer : bool = False
    freeze_old_embedding_weights : bool = False
    freeze_old_regressor_weights : bool = False

    plateau_early_stopping : bool = False
    plateau_early_stopping_patience : int = 0
    plateau_early_stopping_min_delta : float = 0

    inflection_early_stopping : bool = False
    inflection_early_stopping_min_improvement : float = 0
    inflection_early_stopping_patience : int = 0
    inflection_early_stopping_ignore_first_n : int = 0

    inflection_lr_reducer : bool = False
    inflection_lr_reducer_factor : float = 0
    inflection_lr_reducer_min_improvement : float = 0
    inflection_lr_reducer_patience : int = 0

    lr_warmup : bool = False
    lr_warmup_num_steps : int = 0
    lr_warmup_start_lr : float = 0


class AutomaticRlTlTraining:
    config : AutomaticRlTlTrainingConfig

    model : PrositIntensityPredictor
    is_new_model : bool

    requires_new_embedding_layer : bool
    can_reuse_old_embedding_weights : bool
    requires_new_regressor_layer : bool
    can_reuse_old_regressor_weights : bool
    
    current_epoch_offset : int = 0
    callbacks : list = []
    training_schedule : list = []
    validation_steps : Optional[int]

    def __init__(self, config : AutomaticRlTlTrainingConfig):
        """Automatic refinement/transfer learning given a dataset and optionally an existing model. The training process consists of the following phases:
        
        Phase 1:
            This phase is only performed, if new weights were added to the model in the embedding or regressor layer (extended embedding or additional ions). Only the new weights are trained while all other weights are frozen. The training process starts with a learning rate warmup. The phase automatically stops as soon as no major improvements are detected anymore.

        Phase 2:
            This phase resembles the main training process. All weights are trained and no freezing is applied. The phase starts with a learning rate warmup and automatically stops as soon as no major improvements are detected anymore.
        
        Phase 3:
            Optional finetuning phase that is only performed if config.improve_further is set to True. This phase starts with a slightly lower learning rate than the one used in phase 2 and reduces the learning rate when as no significant improvement can be detected anymore. The phase stops automatically as soon as no improvements are detected over a longer period.

        Args:
            config (AutomaticRlTlTrainingConfig): Contains all relevant configuration parameters for performing the automatic refinement/transfer learning process. Please refer to the documentation of AutomaticRlTlTrainingConfig for further documentation.
        """
        self.config = config

        self._init_wandb()
        self._init_model()
        self._update_model_inputs()
        self._update_model_outputs()
        self._init_training()
        self._construct_training_schedule()
    
    def _init_wandb(self):
        """ Initializes Weights & Biases Logging if the user requested that in the config.
        """
        if self.config.use_wandb:
            global wandb
            global WandbCallback
            import wandb
            from wandb.integration.keras import WandbCallback

            wandb.init(
                project=self.config.wandb_project,
                config=self.config.to_dict(),
                tags=self.config.wandb_tags
            )

        
    def _init_model(self):
        """Configures the given baseline model or creates a new model if no baseline model is provided in the config.
        """
        if self.config.baseline_model is not None:
            self.model = self.config.baseline_model
            self.is_new_model = False
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
                seq_length=self.config.dataset.max_seq_len,
                alphabet=self.config.dataset.alphabet,
                use_prosit_ptm_features=False,
                with_termini=False,
                input_keys=input_mapping,
                meta_data_keys=meta_data_keys
            )
            self.is_new_model = True

    def _update_model_inputs(self):
        """Modifies the model's embedding layer to fit the provided dataset. All decisions here are made automatically based on the provided model and dataset.
        """
        model_alphabet = self.model.alphabet
        dataset_alphabet = self.config.dataset.alphabet

        if self.is_new_model:
            print('[embedding layer]  created new model with fresh embedding layer')
            self.requires_new_embedding_layer = False
            self.can_reuse_old_embedding_weights = False
            return

        if model_alphabet == dataset_alphabet:
            print('[embedding layer]  model and dataset modifications match')
            self.requires_new_embedding_layer = False
            self.can_reuse_old_embedding_weights = False
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
        """Modifies the model's regressor layer to fit the provided dataset. All decisions here are made automatically based on the provided model and dataset.

        Raises:
            RuntimeError: Error is raised if the model and the dataset have a different sequence length. A mismatch in the sequence length is not supported.
        """
        # check that sequence length matches
        if self.model.seq_length != self.config.dataset.max_seq_len:
            raise RuntimeError(f"Max. sequence length does not match between dataset and model (dataset: {self.config.dataset.max_seq_len}, model: {self.model.seq_length})")

        if self.is_new_model:
            print('[regressor layer]  created new model with fresh regressor layer')
            self.requires_new_regressor_layer = False
            self.can_reuse_old_regressor_weights = False
            return

        # check whether number of ions matches
        model_ions = ['y', 'b']
        if hasattr(self.model, 'ion_types') and self.model.ion_types is not None:
            model_ions = self.model.ion_types 

        dataset_ions = ['y', 'b']
        if hasattr(self.config.dataset, 'ion_types') and self.config.dataset.ion_types is not None:
            dataset_ions = self.config.dataset.ion_types 

        if model_ions == dataset_ions:
            print('[regressor layer]  matching ion types')
            self.requires_new_regressor_layer = False
            self.can_reuse_old_regressor_weights = False
        else:
            print('[regressor layer]  ion types not matching')
            self.requires_new_regressor_layer = True

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
        """Configures relevant training settings that are used across all phases of the training.
        """
        if self.config.use_wandb:
            class LearningRateReporter(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, *args):
                    wandb.log({'learning_rate': self.model.optimizer.lr.read_value()})

            class RealEpochReporter(tf.keras.callbacks.Callback):
                def on_epoch_begin(self_inner, epoch, *args):
                    wandb.log({'epoch_total': epoch + self.current_epoch_offset})

            self.callbacks = [WandbCallback(save_model=False, log_batch_frequency=True, verbose=1), LearningRateReporter(), RealEpochReporter()]
        

        num_val_batches = self.config.dataset.tensor_val_data.cardinality().numpy()
        self.validation_steps = 1000 if num_val_batches > 1000 else None

    def _evaluate_model(self):
        """Runs an evaluation over max. 1000 batches of the validation set and logs the validation performance.
        """
        loss, metric = self.model.evaluate(
            self.config.dataset.tensor_val_data,
            steps=self.validation_steps
        )

        print(f'validation loss: {loss}, pearson distance: {metric}')
        if self.config.use_wandb:
            wandb.log({'val_loss': loss, 'val_masked_pearson_correlation_distance': metric})

    def _construct_training_schedule(self):
        """Configures the phases of the training process based on the given config and the provided dataset and model.
        """
        self.training_schedule = []

        num_train_batches = self.config.dataset.tensor_train_data.cardinality().numpy()
        batch_size = self.config.dataset.batch_size 
        num_train_sequences = batch_size * num_train_batches 

        is_transfer_learning = self.requires_new_embedding_layer or self.requires_new_regressor_layer

        # step 1:
        #   warm up new weights in embedding/regressor layer
        if is_transfer_learning:
            warmup_sequences = self.config.min_warmup_sequences_new_weights
            warmup_epochs = math.ceil(warmup_sequences / num_train_sequences)
            warmup_batches = math.ceil(warmup_sequences / batch_size)
            training_epochs = 10000
            self.training_schedule.append(TrainingInstanceConfig(
                num_epochs=warmup_epochs + training_epochs,
                learning_rate=1e-3,
                lr_warmup=True,
                lr_warmup_num_steps=warmup_batches,
                lr_warmup_start_lr=1e-8,
                inflection_early_stopping=True,
                inflection_early_stopping_min_improvement=1e-4,
                inflection_early_stopping_ignore_first_n=warmup_batches,
                inflection_early_stopping_patience=1000,
                freeze_inner_layers=True,
                freeze_whole_embedding_layer=not self.requires_new_embedding_layer,
                freeze_whole_regressor_layer=not self.requires_new_regressor_layer,
                freeze_old_embedding_weights=self.requires_new_embedding_layer and self.can_reuse_old_embedding_weights, 
                freeze_old_regressor_weights=self.requires_new_regressor_layer and self.can_reuse_old_regressor_weights 
            ))

        # step 2:
        #   warmup whole model and do main fitting process
        warmup_sequences = self.config.min_warmup_sequences_whole_model
        warmup_epochs = math.ceil(warmup_sequences / num_train_sequences)
        warmup_batches = math.ceil(warmup_sequences / batch_size)
        training_epochs = 10000
        self.training_schedule.append(TrainingInstanceConfig(
            num_epochs=warmup_epochs + training_epochs,
            learning_rate=1e-3,
            lr_warmup=True,
            lr_warmup_num_steps=warmup_batches,
            lr_warmup_start_lr=1e-8,
            inflection_early_stopping=True,
            inflection_early_stopping_min_improvement=1e-5,
            inflection_early_stopping_ignore_first_n=warmup_batches,
            inflection_early_stopping_patience=2000
        ))

        # step 3:
        #   optional: refine the model further to get a really good model
        if self.config.improve_further:
            training_epochs = 10000
            self.training_schedule.append(TrainingInstanceConfig(
                num_epochs=training_epochs,
                learning_rate=1e-4,
                inflection_early_stopping=True,
                inflection_early_stopping_min_improvement=1e-6,
                inflection_early_stopping_ignore_first_n=0,
                inflection_early_stopping_patience=10000,
                inflection_lr_reducer=True,
                inflection_lr_reducer_factor=0.5,
                inflection_lr_reducer_min_improvement=1e-5,
                inflection_lr_reducer_patience=7000
            ))


    def train(self):
        """Performs the training process and returns the final model.

        Returns:
            PrositIntensityPredictor: The refined model that results from the training process. This model can be used for predictions or further training steps.
        """
        self._evaluate_model()

        for instance_config in self.training_schedule:
            training = AutomaticRlTlTrainingInstance(
                instance_config=instance_config,
                model=self.model,
                dataset=self.config.dataset,
                current_epoch_offset=self.current_epoch_offset,
                wandb_logging=self.config.use_wandb,
                callbacks=self.callbacks,
                validation_steps=self.validation_steps
            )
            training.run()

            self.current_epoch_offset = training.current_epoch_offset
            self._evaluate_model()

        if self.config.use_wandb:
            wandb.finish()

        return self.model



class AutomaticRlTlTrainingInstance:

    model : PrositIntensityPredictor
    dataset : FragmentIonIntensityDataset
    instance_config : TrainingInstanceConfig
    current_epoch_offset : int
    wandb_logging : bool
    callbacks : list
    validation_steps : Optional[int]

    stopped_early : bool
    final_learning_rate : float
    inflection_early_stopping : Optional[InflectionPointEarlyStopping] = None


    def __init__(
            self,
            instance_config : TrainingInstanceConfig,
            model : PrositIntensityPredictor,
            current_epoch_offset : int,
            dataset : FragmentIonIntensityDataset,
            wandb_logging : bool,
            callbacks : list,
            validation_steps : Optional[int]
        ):
        self.instance_config = instance_config
        self.model = model
        self.dataset = dataset
        self.current_epoch_offset = current_epoch_offset
        self.wandb_logging = wandb_logging
        self.callbacks = callbacks.copy()
        self.validation_steps = validation_steps

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
                    'freeze_inner_layers': 1,
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
                    'freeze_inner_layers': 0,
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
            self.inflection_early_stopping = InflectionPointEarlyStopping(
                min_improvement=self.instance_config.inflection_early_stopping_min_improvement,
                patience=self.instance_config.inflection_early_stopping_patience,
                ignore_first_n=self.instance_config.inflection_early_stopping_ignore_first_n,
                wandb_log=self.wandb_logging
            )
            
            self.callbacks.append(self.inflection_early_stopping)


        if self.instance_config.inflection_lr_reducer:
            reduce_lr = InflectionPointLRReducer(
                factor=self.instance_config.inflection_lr_reducer_factor,
                patience=self.instance_config.inflection_lr_reducer_patience,
                min_improvement=self.instance_config.inflection_lr_reducer_min_improvement,
                wandb_log=self.wandb_logging
            ) 

            self.callbacks.append(reduce_lr)

        if self.instance_config.lr_warmup:
            lr_warmup_linear = LearningRateWarmupPerStep(
                num_steps=self.instance_config.lr_warmup_num_steps,
                start_lr=self.instance_config.lr_warmup_start_lr,
                end_lr=self.instance_config.learning_rate
            )
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
            validation_steps=self.validation_steps,
            epochs=self.instance_config.num_epochs,
            callbacks=self.callbacks
        )

        inflection_ES_stopped = self.inflection_early_stopping is not None and self.inflection_early_stopping.stopped_early
        if len(history.history['loss']) < self.instance_config.num_epochs or inflection_ES_stopped:
            self.stopped_early = True
        else:
            self.stopped_early = False

        self.final_learning_rate = self.model.optimizer._learning_rate.numpy()
        self.current_epoch_offset += len(history.history['loss'])

