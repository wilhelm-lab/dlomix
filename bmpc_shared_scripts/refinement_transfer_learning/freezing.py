import dlomix
import tensorflow as tf
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance




# function to freeze all layers except first and/or last layer
def freeze_model(model:dlomix.models.prosit.PrositIntensityPredictor,optimizer:tf.keras.optimizers, trainable_first_layer:bool = False, trainable_last_layer:bool = False, loss:dlomix.losses=masked_spectral_distance, metrics:list=[masked_pearson_correlation_distance]) -> None:
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