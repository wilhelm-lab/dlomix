import yaml
import os
import uuid
import wandb
from wandb.integration.keras import WandbCallback
import keras
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,
    LambdaCallback, TerminateOnNaN, CSVLogger
)
from dlomix.data import FragmentIonIntensityDataset, load_processed_dataset
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.models import PrositIntensityPredictor
from dlomix.constants import PTMS_ALPHABET


def early_stopping_callback(monitor: str, min_delta: float, patience: int, restore_best_weights: bool) -> EarlyStopping:
    """
    Creates an EarlyStopping callback.

    Args:
    monitor (str): Quantity to be monitored.
    min_delta (float): Minimum change to qualify as an improvement.
    patience (int): Number of epochs with no improvement after which training will be stopped.
    restore_best_weights (bool): Whether to restore model weights from the epoch with the best value of the monitored quantity.

    Returns:
    tensorflow.keras.callbacks.EarlyStopping: The EarlyStopping callback.
    """
    return EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        restore_best_weights=restore_best_weights
    )

def reduce_lr_callback(monitor: str, factor: float, patience: int, min_lr: float) -> ReduceLROnPlateau:
    """
    Creates a ReduceLROnPlateau callback.

    Args:
    monitor (str): Quantity to be monitored.
    factor (float): Factor by which the learning rate will be reduced.
    patience (int): Number of epochs with no improvement after which learning rate will be reduced.
    min_lr (float): Lower bound on the learning rate.

    Returns:
    tensorflow.keras.callbacks.ReduceLROnPlateau: The ReduceLROnPlateau callback.
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )

def learning_rate_scheduler_callback(initial_lr: float, decay_rate: float) -> LearningRateScheduler:
    """
    Creates a LearningRateScheduler callback.

    Args:
    initial_lr (float): Initial learning rate.
    decay_rate (float): Decay rate for the learning rate.

    Returns:
    tensorflow.keras.callbacks.LearningRateScheduler: The LearningRateScheduler callback.
    """
    return LearningRateScheduler(
        schedule=lambda epoch: initial_lr * decay_rate ** epoch
    )

def terminate_on_nan_callback() -> TerminateOnNaN:
    """
    Creates a TerminateOnNaN callback.

    Returns:
    tensorflow.keras.callbacks.TerminateOnNaN: The TerminateOnNaN callback.
    """
    return TerminateOnNaN()

def lambda_callback() -> LambdaCallback:
    """
    Creates a LambdaCallback to print epoch start.

    Returns:
    tensorflow.keras.callbacks.LambdaCallback: The LambdaCallback.
    """
    return LambdaCallback(
        on_epoch_begin=lambda epoch, logs: print(f"Starting epoch {epoch + 1}")
    )

def csv_logger_callback(filename: str, append: bool) -> CSVLogger:
    """
    Creates a CSVLogger callback.

    Args:
    filename (str): Filename of the CSV file.
    append (bool): Whether to append if file exists.

    Returns:
    tensorflow.keras.callbacks.CSVLogger: The CSVLogger callback.
    """
    return CSVLogger(
        filename=filename, 
        append=append
    )

def model_checkpoint_callback(filepath: str, monitor: str, save_best_only: bool, save_weights_only: bool, mode: str, save_freq: str, verbose: int) -> ModelCheckpoint:
    """
    Creates a ModelCheckpoint callback.

    Args:
    filepath (str): Path to save the model file.
    monitor (str): Quantity to be monitored.
    save_best_only (bool): If True, the latest best model will not be overwritten.
    save_weights_only (bool): If True, then only the model's weights will be saved.
    mode (str): One of {'auto', 'min', 'max'}.
    save_freq (str or int): Interval (number of epochs or batches) between checkpoints.
    verbose (int): Verbosity mode.

    Returns:
    tensorflow.keras.callbacks.ModelCheckpoint: The ModelCheckpoint callback.
    """
    return ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode,
        save_freq=save_freq,
        verbose=verbose
    )


# Manually specify the path to the configuration file (Note: change path according to your directories)
config_file_path = '/nfs/home/students/s.baier/mapra/dlomix/bmpc_shared_scripts/refinement_transfer_learning/config_files/baseline_noptm_baseline_small_bs1024.yaml'

with open(config_file_path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Show config containing the configuration data
print(config)

# configure environment
os.environ['HF_HOME'] = config['dataset']['hf_home']
os.environ['HF_DATASETS_CACHE'] = config['dataset']['hf_cache']

# set id for run using uuid
config['run_id'] = uuid.uuid4()

# set up wandb for this project
project_name = f'callback model training'
wandb.init(
    project=project_name,
    config=config,
    tags=[config['dataset']['name']], 
    entity='mapra_dlomix'
)

# DLOmix dataset 
dataset = load_processed_dataset(wandb.config['dataset']['processed_path'])

# Initialize TensorFlow and the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config['training']['learning_rate'])

# Initialize callbacks
early_stopping = early_stopping_callback(
    monitor=config['callbacks']['early_stopping']['monitor'],
    min_delta=config['callbacks']['early_stopping']['min_delta'],
    patience=config['callbacks']['early_stopping']['patience'],
    restore_best_weights=config['callbacks']['early_stopping']['restore_best_weights']
)

reduce_lr = reduce_lr_callback(
    monitor=config['callbacks']['reduce_lr']['monitor'],
    factor=config['callbacks']['reduce_lr']['factor'],
    patience=config['callbacks']['reduce_lr']['patience'],
    min_lr=config['callbacks']['reduce_lr']['min_lr']
)

learning_rate_scheduler = learning_rate_scheduler_callback(
    initial_lr=config['callbacks']['learning_rate_scheduler']['initial_lr'],
    decay_rate=config['callbacks']['learning_rate_scheduler']['decay_rate']
)

terminate_on_nan = terminate_on_nan_callback()
lambda_cb = lambda_callback()

csv_logger = csv_logger_callback(
    filename=config['callbacks']['csv_logger']['filename'],
    append=config['callbacks']['csv_logger']['append']
)

model_checkpoint = model_checkpoint_callback(
    filepath=config['callbacks']['model_checkpoint']['filepath'],
    monitor=config['callbacks']['model_checkpoint']['monitor'],
    save_best_only=config['callbacks']['model_checkpoint']['save_best_only'],
    save_weights_only=config['callbacks']['model_checkpoint']['save_weights_only'],
    mode=config['callbacks']['model_checkpoint']['mode'],
    save_freq=config['callbacks']['model_checkpoint']['save_freq'],
    verbose=config['callbacks']['model_checkpoint']['verbose']
)

# Collect all callbacks in a list
callbacks = [
    WandbCallback(save_model=False, log_batch_frequency=True),
    early_stopping,
    reduce_lr,
    learning_rate_scheduler,
    terminate_on_nan,
    lambda_cb,
    csv_logger,  # (Note: not necessary when using wandb; shown for completeness)
    model_checkpoint
]

input_mapping = {
    "SEQUENCE_KEY": "modified_sequence",
    "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
    "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
    "FRAGMENTATION_TYPE_KEY": "method_nbr",
}

meta_data_keys = ["collision_energy_aligned_normed", "precursor_charge_onehot", "method_nbr"]

# initialize prosit model
model = PrositIntensityPredictor(
    seq_length=wandb.config['dataset']['seq_length'],
    alphabet=PTMS_ALPHABET,
    use_prosit_ptm_features=False,
    with_termini=False,
    input_keys=input_mapping,
    meta_data_keys=meta_data_keys
)

# Compile the model 
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
    callbacks=callbacks
)

# model path to save to the model to (Note: The file needs to end with the .keras extension.)
model_path = f"{wandb.config['model']['save_dir']}/{wandb.config['dataset']['name']}/{wandb.config['run_id']}.keras"

# save the model
model.save(model_path)  

print(f"Model saved to: {model_path}")

# Finish the wandb run
wandb.finish()

# load the trained model 
reconstructed_model = keras.models.load_model(model_path)

# Model summary 

# Print parameters
print("Embedding Output Dimension:", reconstructed_model.embedding_output_dim)
print("Sequence Length:", reconstructed_model.seq_length)
print("Alphabet Dictionary:", reconstructed_model.alphabet)
print("Dropout Rate:", reconstructed_model.dropout_rate)
print("Latent Dropout Rate:", reconstructed_model.latent_dropout_rate)
print("Recurrent Layers Sizes:", reconstructed_model.recurrent_layers_sizes)
print("Regressor Layer Size:", reconstructed_model.regressor_layer_size)
print("Use Prosit PTM Features:", reconstructed_model.use_prosit_ptm_features)
print("Input Keys:", reconstructed_model.input_keys)

# Print attributes
print("Default Input Keys:", reconstructed_model.DEFAULT_INPUT_KEYS)
print("Meta Data Keys (Attribute):", reconstructed_model.META_DATA_KEYS)
print("PTM Input Keys:", reconstructed_model.PTM_INPUT_KEYS)
