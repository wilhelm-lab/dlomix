from wandb.integration.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau,
    LambdaCallback, TerminateOnNaN, CSVLogger
)

import dlomix
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.models import PrositIntensityPredictor



def change_output_layer(model: PrositIntensityPredictor, number_of_ions: int = 2) -> None:
    """
    Change the output layer of a PrositItensityPredictor model
    This means changing the number of predicted ions to the number of ion types in the dataset.
    The default PrositIntensityPredictor predicts two ion types (y- and b-ions). 
    If the number of ions is not given, this function will replace the output layer with a randomly initialized layer the same dimensions as before.

    If the number of ions changes to for example 4, the regressor will have an output dimension of:
        (batch_size, number_of_ions * charge_states * possible ions) = (batch_size, 4 * 3 * 29) = (batch_size, 348)
    After changing the output layer, the models needs to be compiled again before training.

    Args:
        model (PrositIntensityPredictor): the model where the output layer changes
        number_of_ions (int, optional): Number of ions the model should be able to predict. Defaults to 2.
    """
    model.len_fion = 3 * number_of_ions
    model.regressor = tf.keras.Sequential(
        [
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(model.len_fion), name='time_dense'
                ), 
            tf.keras.layers.LeakyReLU(name='activation'), 
            tf.keras.layers.Flatten(name='out')], 
        name='regressor'
        )


def change_input_layer(model: PrositIntensityPredictor, modifications: list = None) -> None:
    """Change the input layer of a PrositIntensityPredictor model
    This means changing the number of embeddings the Embedding layer can produce. This is directly tied to the size of the alphabet of the model.
    A list of new modifications the model should support is given and the modifications are added to the alphabet, increasing its size.
    If no new modifications are given, the weights for the Embedding layer are re-initialized.
    After changing the input layer, the models needs to be compiled again before training.

    Args:
        model (PrositIntensityPredictor): The model, where the input layers needs to be changed
        modifications (list, optional): List of modifications the model should support. Defaults to None.
    """
    if modifications:
        for new_mod in modifications:
            model.alphabet.update({new_mod: max(model.alphabet.values()) + 1})

    model.embedding = tf.keras.layers.Embedding(
        input_dim=len(model.alphabet) + 2,
        output_dim=model.embedding_output_dim,
        input_length=model.seq_length,
        name='embedding'
    )


def freeze_model(model: PrositIntensityPredictor, optimizer:tf.keras.optimizers, trainable_first_layer:bool = False, trainable_last_layer:bool = False, loss:dlomix.losses=masked_spectral_distance, metrics:list=[masked_pearson_correlation_distance]) -> None:
    ''' Freezes all layers of a PrositIntensityPredictor and keep first and/or last layer trainable.

    First setting the whole model to trainable because this attribute overshadows the trainable attribute of every sublayer.
    Then iterating through all sublayers and sets the trainable attribute of every layer to 'False', model is now frozen.
    Next, setting the trainable attribute of either the first embedding layer or the last time density layer to trainable.
    Finally, compile the model with the optimizer, loss, and metrics to make the changes take effect.

    Parameter
    ---------
    model                   : dlomix.models.prosit.PrositIntensityPredictor
                              The model to be frozen.
    optimizer               : tf.keras.optimizers
                              The optimizer is needed for compiling the model.
    trainable_first_layer   : bool
                              Whether the first layer should remain trainable.
    trainable_last_layer    : bool
                              Whether the last layer should remain trainable
    loss                    : dlomix.losses
                              The loss for compiling the model. 
                              default: masked_spectral_distance
    metrics                 : list[dlomix.losses]
                              The metrics for compiling the model.
                              default: [masked_pearson_correlation_distance] 
    --------

    '''

    model.trainable = True 
    for lay in model.layers:
        try:
            for sublay in lay.layers:
                sublay.trainable = False
        except (AttributeError):
            lay.trainable = False

    if (trainable_first_layer):
        first_layer = model.get_layer(name="embedding")
        first_layer.trainable = True

    if (trainable_last_layer):
        last_layer = model.get_layer(name = "sequential_4").get_layer(name = "time_dense")
        last_layer.trainable = True

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
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


# Dummy to be adjusted for specific usecases
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
