from wandb.integration.keras import WandbCallback
import tensorflow
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,
    LambdaCallback, TerminateOnNaN, CSVLogger
)


def wandb_callback(save_model: bool, log_batch_frequency: int, log_weights: bool) -> WandbCallback:
    """
    Creates a WandbCallback for use with model training.

    Args:
    save_model (bool): Whether to save the model at the end of every epoch. It requires `val_data` to be part of fit.
    log_batch_frequency (int): Frequency (in number of batches) at which to log training data. Use None to disable.
    log_weights (bool): Whether to log model weights to wandb.

    Returns:
    WandbCallback: The configured WandbCallback.
    """
    return WandbCallback(
        save_model=save_model, 
        log_batch_frequency=log_batch_frequency,
        log_weights=log_weights
    )

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
