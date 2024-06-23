import dlomix




# function to freeze all layers except first and/or last layer
def freeze_model(model:dlomix.models.prosit.PrositIntensityPredictor, trainable_first_layer:bool = False, trainable_last_layer:bool = False) -> None:
    ''' Freezes all layers of a PrositIntensityPredictor and keep first and/or last layer trainable.

    First setting the whole model to trainable because this attribute overshadows the trainable attribute of every sublayer.
    Then iterating through all sublayers and sets the trainable attribute of every layer to 'False', model is now frozen.
    Next, setting the trainable attribute of either the first embedding layer or the last time density layer to trainable.

    Parameter
    ---------
    model                   : dlomix.models.prosit.PrositIntensityPredictor
                              The model to be frozen.
    trainable_first_layer   : bool
                              Whether the first layer should remain trainable.
    trainable_last_layer    : bool
                              Whether the last layer should remain trainable
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
