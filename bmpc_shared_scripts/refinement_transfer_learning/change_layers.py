import tensorflow as tf
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
            )
            tf.keras.layers.LeakyReLU(name='activation'),
            tf.keras.layers.Flatten(name='out')
        ], 
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
