import tensorflow as tf
from dlomix.models import PrositIntensityPredictor
from tensorflow.keras.constraints import Constraint
import keras.backend as K
import keras
from dlomix.models import PrositIntensityPredictor


@keras.saving.register_keras_serializable()
class FixRegressorWeights(Constraint):
    def __init__(self, old_weights, old_fions):
        self.old_weights = old_weights
        self.freeze_weights = True
        self.old_fions = old_fions
    def __call__(self, w):
        if self.freeze_weights:
            return K.concatenate([self.old_weights, w[:, self.old_fions:]], axis=1)
        return w


@keras.saving.register_keras_serializable()
class FixBias(Constraint):
    def __init__(self, old_bias, old_fions):
        self.old_bias = old_bias
        self.freeze_bias = True
        self.old_fions = old_fions
    def __call__(self, b):
        if self.freeze_bias:
            return K.concatenate([self.old_bias, b[self.old_fions:]], axis=0)
        return b


def change_output_layer(model: PrositIntensityPredictor, number_of_ions: int = 2, freeze_old_output: bool = False) -> None:
    """
    Change the output layer of a PrositItensityPredictor model
    This means changing the number of predicted ions to the number of ion types in the dataset.
    The default PrositIntensityPredictor predicts two ion types (y- and b-ions). 
    If the number of ions is not given, this function will replace the output layer with a randomly initialized layer the same dimensions as before.

    If the number of ions changes to for example 4, the regressor will have an output dimension of:
        (batch_size, number_of_ions * charge_states * possible ions) = (batch_size, 4 * 3 * 29) = (batch_size, 348)
    After changing the output layer, the models needs to be compiled again before training.

    It is possible to fix the old weights and the old bias of the regressor layer before reinitializing the regressor layer. 
    To do so, set the freeze_old_output to 'True'

    Args:
        model (PrositIntensityPredictor): the model where the output layer changes
        number_of_ions (int, optional): Number of ions the model should be able to predict. Defaults to 2.
        freeze_old_output (bool, optional): Specify if the pre-trained regressor weight should be kept in place if reinitializing the embedding layer
    """
    kernel_constraint = None
    bias_constraint = None
    if freeze_old_output:
        old_weights = model.regressor.get_layer('time_dense').get_weights()[0]
        old_bias = model.regressor.get_layer('time_dense').get_weights()[1]
        
        kernel_constraint = FixRegressorWeights(old_weights, model.len_fion)
        bias_constraint = FixBias(old_bias, model.len_fion)

    model.len_fion = 3 * number_of_ions
    model.regressor = tf.keras.Sequential(
        [
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    model.len_fion,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint
                    ), name='time_dense'
                ), 
            tf.keras.layers.LeakyReLU(name='activation'), 
            tf.keras.layers.Flatten(name='out')], 
        name='regressor'
        )


def release_old_regressor(model: PrositIntensityPredictor):
    """Function to release the pre-trained regressor of a re-initialized regressor layer of the Prosit model
    The freeze_weights parameter changes the constraint, so that the weights do not get overwritten by the old weights
    In theory, the regressor weights and bias can be frozen again.

    Args:
        model (PrositIntensityPredictor): the model where to release the regressor
    """
    if model.regressor.get_layer('time_dense').kernel_constraint is not None:
        model.regressor.get_layer('time_dense').kernel_constraint.freeze_weights = False
        model.regressor.get_layer('time_dense').bias_constraint.freeze_weights = False


@keras.saving.register_keras_serializable()
class FixWeights(Constraint):
    def __init__(self, old_weights, max_old_embedding):
        self.old_weights = old_weights
        self.freeze_weights = True
        self.max_embedding_value = max_old_embedding
    def __call__(self, w):
        if self.freeze_weights:
            return K.concatenate([self.old_weights[:self.max_embedding_value + 1], w[self.max_embedding_value + 1:]], axis=0)
        return w
    

def change_input_layer(model: PrositIntensityPredictor, modifications: list = None, freeze_old_embeds: bool = False) -> None:
    """Change the input layer of a PrositIntensityPredictor model
    This means changing the number of embeddings the Embedding layer can produce. This is directly tied to the size of the alphabet of the model.
    A list of new modifications the model should support is given and the modifications are added to the alphabet, increasing its size.
    If no new modifications are given, the weights for the Embedding layer are re-initialized.

    This function also allows the user to freeze the old embedding weights trained by the loaded model,
    meaning it only allows changing the weights for the embeddings of the new modifications.

    After changing the input layer, the models needs to be compiled again before training.

    Args:
        model (PrositIntensityPredictor): The model, where the input layers needs to be changed
        modifications (list, optional): List of modifications the model should support. Defaults to None.
        freeze_old_embeds (bool): If set to True, the old embeddings of the loaded model are not changed during training. Defaults to False.
    """
    old_embedding_max = max(model.alphabet.values())
    if modifications:
        model.alphabet.update({k: i for i, k in enumerate(modifications, start=len(model.alphabet) + 1)})
        
    embeddings_constraint = None
    if freeze_old_embeds:
        # if added names to the model, replace get_layer index with name 
        trained_embeds_weights = model.layers[0].get_weights()[0]
        embeddings_constraint = FixWeights(trained_embeds_weights, max_old_embedding=old_embedding_max)

    model.embedding = tf.keras.layers.Embedding(
        input_dim=len(model.alphabet) + 2,
        output_dim=model.embedding_output_dim,
        input_length=model.seq_length,
        embeddings_constraint=embeddings_constraint,
        name='embedding'
        )

def release_old_embeddings(model: PrositIntensityPredictor):
    """Function to release the pre-trained embeddings of a re-initialized embedding layer of the Prosit model
    The freeze_weights parameter changes the constraint, so that the weights do not get overwritten by the old weights
    In theory, the embeddings can be frozen again.

    Args:
        model (PrositIntensityPredictor): model with a changed embedding layer named 'embedding'
    """
    if model.get_layer('embedding').embeddings_constraint is not None:
        model.get_layer('embedding').embeddings_constraint.freeze_weights = False
