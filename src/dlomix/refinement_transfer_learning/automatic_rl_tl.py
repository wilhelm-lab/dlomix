import os
import sys

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, CSVLogger

from .custom_callbacks import CustomCSVLogger, BatchEvaluationCallback, InflectionPointEarlyStopping, LearningRateWarmupPerStep, InflectionPointLRReducer, OverfittingEarlyStopping

from dlomix.constants import PTMS_ALPHABET, ALPHABET_NAIVE_MODS, ALPHABET_UNMOD
from dlomix.data import load_processed_dataset, FragmentIonIntensityDataset
from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.refinement_transfer_learning import change_layers, freezing

from dataclasses import dataclass, asdict, field
from typing import Optional
import math
import json
import numpy as np
from pathlib import Path
import importlib.resources as importlib_resources
import shutil

from nbconvert import HTMLExporter
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


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

    # csv logger parameters
    results_log : str = 'results_log'

    
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
    results_data_path : Path
    results_notebook_path : Path 

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

    initial_loss : float = None

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
        self._init_logging()
        self._init_model()
        self._update_model_inputs()
        self._update_model_outputs()
        self._init_training()
        self._construct_training_schedule()
        self._explore_data()
    
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

    def _init_logging(self):
        """ Initializes Weights & Biases Logging and CSV Logging if the user requested that in the config.
        """       
        
        self.results_data_path = Path(self.config.results_log) / 'log_data/'

        if not os.path.exists(self.results_data_path):
            os.makedirs(self.results_data_path)

        notebook_ref = importlib_resources.files('dlomix') / 'refinement_transfer_learning' / 'user_report.ipynb'
        self.results_notebook_path = Path(self.config.results_log) / 'report.ipynb'
        with importlib_resources.as_file(notebook_ref) as path: 
            shutil.copyfile(path, self.results_notebook_path)

        self.csv_logger = CustomCSVLogger(f'{self.results_data_path}/training_log.csv', separator=',', append=True)

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
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(
            optimizer=optimizer,
            loss=masked_spectral_distance,
            metrics=[masked_pearson_correlation_distance]
        )
    

    def _calculate_spectral_angles(self, stage):
        """Calculates and saves the spectral angle distributions before and after training."""

        def calculate_spectral_distance(dataset, model, max_batches=1000):
            spectral_dists = []
            for i, (batch, y_true) in enumerate(dataset):
                if i >= max_batches:
                    break
                y_pred = model.predict(batch)
                spectral_dists.extend(masked_spectral_distance(y_true=y_true, y_pred=y_pred).numpy())
            return spectral_dists

        def calculate_and_save_spectral_angle_distribution(data, model, results_log, stage, datasets=['train', 'val', 'test']):
            """
            Predict the intensities, calculate spectral distances, and save the spectral angle distribution for the specified datasets.

            Args:
                data: A dataset containing tensor_train_data, tensor_val_data, and tensor_test_data.
                model: A trained model used for making predictions.
                results_log: Directory to save the JSON files.
                stage: A string indicating the stage ('before' or 'after').
                datasets: A list of strings indicating which datasets to use ('train', 'val', 'test').

            Returns:
                None (saves JSON files)
            """
            def save_json(data, filename):
                with open(os.path.join(results_log, filename), 'w') as f:
                    json.dump(data, f)

            for dataset in datasets:
                if dataset not in ['train', 'val', 'test']:
                    raise ValueError("Invalid dataset type. Choose 'train', 'val', or 'test'.")

                try:
                    if dataset == 'train':
                        dataset_data = data.tensor_train_data
                    elif dataset == 'val':
                        dataset_data = data.tensor_val_data
                    elif dataset == 'test':
                        dataset_data = data.tensor_test_data
                except ValueError:
                    continue
                

                spectral_dists = calculate_spectral_distance(dataset_data, model)
                sa_data = [1 - sd for sd in spectral_dists]
                avg_sa = np.mean(sa_data)

                data_to_save = {
                    'spectral_angles': sa_data,
                    'average_spectral_angle': avg_sa
                }

                # Load existing data if present
                filename = f'spectral_angle_distribution_{dataset}.json'
                file_path = os.path.join(results_log, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                else:
                        existing_data = {}

                existing_data[stage] = data_to_save

                save_json(existing_data, filename)

        calculate_and_save_spectral_angle_distribution(
            data=self.config.dataset,
            model=self.model,
            results_log=self.results_data_path,
            stage=stage,
            datasets=['train', 'val', 'test']
        )

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
        self.callbacks = []
        if self.config.use_wandb:
            class LearningRateReporter(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch, *args):
                    wandb.log({'learning_rate': self.model.optimizer.lr.read_value()})

            class RealEpochReporter(tf.keras.callbacks.Callback):
                def on_epoch_begin(self_inner, epoch, *args):
                    wandb.log({'epoch_total': epoch + self.current_epoch_offset})

            self.callbacks = [WandbCallback(save_model=False, log_batch_frequency=True, verbose=1), LearningRateReporter(), RealEpochReporter(), self.csv_logger]
        else:         
            self.callbacks = [             
                self.csv_logger
            ]        

        self.progress_reporter_min_loss = None
        class LossProgressReporter(tf.keras.callbacks.Callback):
            counter : int = 0
            def on_train_batch_end(self_inner, batch, logs):
                loss = logs['loss']

                if self.progress_reporter_min_loss is None:
                    self.progress_reporter_min_loss = loss

                loss = min(self.progress_reporter_min_loss, loss)

                if self_inner.counter % 1000 == 0:
                    approx_progress = min(0.9999, max(0, (self.initial_loss - loss) / (self.initial_loss - 0.1)))
                    print(f'[training]  masked spectral distance: {loss}, approx. progress: {approx_progress * 100:.2f}%', file=sys.stderr)

                self_inner.counter += 1

        self.callbacks.append(LossProgressReporter())

        num_train_batches = self.config.dataset.tensor_train_data.cardinality().numpy()
        batch_size = self.config.dataset.batch_size 
        num_train_sequences = batch_size * num_train_batches 
        self.callbacks.append(OverfittingEarlyStopping(
            max_validation_train_difference=0.1,
            patience=max(2, math.ceil(2000000 / num_train_sequences)),
            wandb_log=self.config.use_wandb
        ))
                
        num_val_batches = self.config.dataset.tensor_val_data.cardinality().numpy()
        self.validation_steps = 1000 if num_val_batches > 1000 else None



    def _evaluate_model(self):
        """Runs an evaluation over max. 1000 batches of the validation set and logs the validation performance.
        """
        loss, metric = self.model.evaluate(
            self.config.dataset.tensor_val_data,
            steps=self.validation_steps,
            verbose=0
        )

        if self.initial_loss is None:
            self.initial_loss = loss

        print(f'validation loss: {loss}, pearson distance: {metric}')
        if self.config.use_wandb:
            wandb.log({'val_loss': loss, 'val_masked_pearson_correlation_distance': metric})
        
        self.csv_logger.set_validation_metrics(val_loss=loss, val_masked_pearson_correlation_distance=metric)
        return {'val_loss': loss, 'val_masked_pearson_correlation_distance': metric}    

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
                learning_rate=1e-4,
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
            learning_rate=1e-4,
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
                inflection_early_stopping_min_improvement=1e-7,
                inflection_early_stopping_ignore_first_n=0,
                inflection_early_stopping_patience=100000,
                inflection_lr_reducer=True,
                inflection_lr_reducer_factor=0.7,
                inflection_lr_reducer_min_improvement=1e-7,
                inflection_lr_reducer_patience=5000
            ))
    def _explore_data(self):
        """Generates and saves exploratory data plots in the results_log folder."""
        def save_json(data, filename):
            with open(os.path.join(self.results_data_path, filename), 'w') as f:
                json.dump(data, f)       

        def plot_amino_acid_distribution(dataset, alphabet, dataset_name):
            """Plots the frequency of each amino acid in the sequences for a given dataset split."""
            def count_amino_acids(sequences):
                aa_counts = {aa: 0 for aa in alphabet}
                for seq in sequences:
                    for aa in seq:
                        if aa in aa_counts:
                            aa_counts[aa] += 1
                return list(aa_counts.values())
            
            sequences = dataset[self.config.dataset.dataset_columns_to_keep[0]]
            aa_counts = count_amino_acids(sequences)
            alphabet_keys = list(alphabet.keys())

            data = {
                'alphabet': alphabet_keys,
                'counts': aa_counts
            }
            save_json(data, f'amino_acid_distribution_{dataset_name}.json')

        def plot_distribution(dataset, feature, dataset_name, transform_func=None, bins=None, xlabel='', ylabel='Frequency', is_sequence=False):
            """General function to plot distributions for different features."""
            feature_data = dataset[feature]
            if transform_func:
                feature_data = transform_func(feature_data)
            if is_sequence:
                feature_data = [len(seq) for seq in feature_data]

            if is_sequence:
                # Define bins to cover the integer range of sequence lengths
                actual_bins = np.arange(min(feature_data) - 0.5, max(feature_data) + 1.5, 1)
            else:
                actual_bins = bins(feature_data) if callable(bins) else bins if bins is not None else 30
        
            hist, bin_edges = np.histogram(feature_data, bins=actual_bins)

            data = {
                'hist': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'xlabel': xlabel,
                'ylabel': ylabel
            }

            if is_sequence: 
                feature = 'sequence'

            save_json(data, f'{feature}_distribution_{dataset_name}.json')

        eval_datasets = {
            'train': self.config.dataset.hf_dataset['train'],
            'val': self.config.dataset.hf_dataset['val'],
            'test': self.config.dataset.hf_dataset['test'] if 'test' in self.config.dataset.hf_dataset else None
        }
                

        for dataset_name, dataset in eval_datasets.items():            
            if dataset:
                if self.config.dataset.dataset_columns_to_keep[0] is not None:
                    plot_amino_acid_distribution(dataset, self.config.dataset.alphabet, dataset_name)
                    plot_distribution(dataset, self.config.dataset.dataset_columns_to_keep[0], dataset_name, is_sequence=True, bins=None, xlabel='Sequence Length')

                plot_distribution(dataset, 'collision_energy_aligned_normed', dataset_name, xlabel='Collision Energy')
                # plot_distribution(dataset, 'intensities_raw', dataset_name, lambda x: [i for sub in x for i in sub], xlabel='Intensity')                
                plot_distribution(dataset, 'precursor_charge_onehot', dataset_name, lambda x: np.argmax(x, axis=1), bins=np.arange(6) - 0.5, xlabel='Precursor Charge')

    def _compile_report(self):
        """Creates a visual PDF report from the jupyter notebook in the results folder
        """
        with open(self.results_notebook_path, 'r') as notebook_file:
            notebook = nbformat.read(notebook_file, as_version=4)

        current_cwd = os.getcwd()
        os.chdir(self.config.results_log)
        executor = ExecutePreprocessor()
        executor.preprocess(notebook)
        os.chdir(current_cwd)

        exporter = HTMLExporter()
        exporter.exclude_input = True
        result, resources = exporter.from_notebook_node(notebook)

        result_path = Path(self.config.results_log) / 'report.html'
        with open(result_path, "w") as f:
            f.write(result)


    def train(self):
        """Performs the training process and returns the final model.

        Returns:
            PrositIntensityPredictor: The refined model that results from the training process. This model can be used for predictions or further training steps.
        """
        self._calculate_spectral_angles('before')
        self._evaluate_model()

        # Add the batch evaluation callback to the callbacks list
        batch_eval_callback = BatchEvaluationCallback(self._evaluate_model, 1000)
        self.callbacks.append(batch_eval_callback)

        for instance_config in self.training_schedule:

            self.csv_logger.reset_phase()

            training = AutomaticRlTlTrainingInstance(
                instance_config=instance_config,
                model=self.model,
                dataset=self.config.dataset,
                current_epoch_offset=self.current_epoch_offset,
                wandb_logging=self.config.use_wandb,
                results_log=self.results_data_path,
                callbacks=self.callbacks,
                validation_steps=self.validation_steps
            )
            training.run()

            self.current_epoch_offset = training.current_epoch_offset
            self._evaluate_model()

        if self.config.use_wandb:
            wandb.finish()
        
        self._calculate_spectral_angles('after')
        self._compile_report()

        return self.model



class AutomaticRlTlTrainingInstance:

    model : PrositIntensityPredictor
    dataset : FragmentIonIntensityDataset
    instance_config : TrainingInstanceConfig
    current_epoch_offset : int
    wandb_logging : bool
    results_log: str 
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
            results_log: str,
            callbacks : list,
            validation_steps : Optional[int]
        ):
        self.instance_config = instance_config
        self.model = model
        self.dataset = dataset
        self.current_epoch_offset = current_epoch_offset
        self.wandb_logging = wandb_logging
        self.results_log = results_log
        self.callbacks = callbacks.copy()
        self.validation_steps = validation_steps

        self._configure_training()


    def _configure_training(self):

        # freezing of old embedding weights
        if self.instance_config.freeze_old_embedding_weights:
            if self.wandb_logging:
                wandb.log({'freeze_old_embedding_weights': 1})
            with open(f'{self.results_log}/freeze_log.csv', 'a') as f:
                f.write('freeze_old_embedding_weights,1\n')
            change_layers.freeze_old_embeddings(self.model)
        else:
            if self.wandb_logging:
                wandb.log({'freeze_old_embedding_weights': 0})
            with open(f'{self.results_log}/freeze_log.csv', 'a') as f:
                f.write('freeze_old_embedding_weights,0\n')
            change_layers.release_old_embeddings(self.model)
        
        # freezing of old regressor weights
        if self.instance_config.freeze_old_regressor_weights:
            if self.wandb_logging:
                wandb.log({'freeze_old_regressor_weights': 1})
            with open(f'{self.results_log}/freeze_log.csv', 'a') as f:
                f.write('freeze_old_regressor_weights,1\n')
            change_layers.freeze_old_regressor(self.model)
        else:
            if self.wandb_logging:
                wandb.log({'freeze_old_regressor_weights': 0})
            with open(f'{self.results_log}/freeze_log.csv', 'a') as f:
                f.write('freeze_old_regressor_weights,0\n')
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
            with open(f'{self.results_log}/freeze_log.csv', 'a') as f:
                f.write(f'freeze_inner_layers,1\nfreeze_embedding_layer,{1 if self.instance_config.freeze_whole_embedding_layer else 0}\nfreeze_regressor_layer,{1 if self.instance_config.freeze_whole_regressor_layer else 0}\n\n')
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
            with open(f'{self.results_log}/freeze_log.csv', 'a') as f:
                f.write(f'freeze_inner_layers,0\nfreeze_embedding_layer,0\nfreeze_regressor_layer,0\n\n')


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
            callbacks=self.callbacks,
            verbose=0
        )

        inflection_ES_stopped = self.inflection_early_stopping is not None and self.inflection_early_stopping.stopped_early
        if len(history.history['loss']) < self.instance_config.num_epochs or inflection_ES_stopped:
            self.stopped_early = True
        else:
            self.stopped_early = False

        self.final_learning_rate = self.model.optimizer._learning_rate.numpy()
        self.current_epoch_offset += len(history.history['loss'])
